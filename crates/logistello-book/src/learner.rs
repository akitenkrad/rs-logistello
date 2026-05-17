//! Opening-book learner: self-play + Negamax back-propagation + drawishness
//! (design doc §4.3.7 / Buro 1999, "Toward Opening Book Learning").
//!
//! Implements the §4.3.7 `LearnOpeningBook(num_games, depth_limit)`
//! pseudocode exactly:
//!
//! 1. `book ← {}` — a [`HashMap`] keyed by the `logistello_core` Zobrist
//!    hash (the book owns one shared key table), each entry an aggregate
//!    over the `(position, move, final_eval)` triples seen in self-play.
//! 2. repeat `num_games`: `game ← SelfPlay(depth_limit)` — every move chosen
//!    by a depth-`depth_limit` search (Phase 2/3 [`decide_move`]); for each
//!    `(position, move, final_eval)` in the game,
//!    `book[position].UpdateStatistics(move, final_eval)`.
//! 3. [`negamax_backpropagate`] — recompute each in-book position's value as
//!    the negamax of its in-book successors' values (a non-in-book or
//!    terminal/frontier position contributes its accumulated self-play /
//!    exact terminal eval). The B6 pass node negates with **no** depth /
//!    empties change (design doc §4.5 B6).
//! 4. [`apply_drawishness`] — shift each position's `best_move` toward
//!    children with higher opponent-error potential
//!    ([`crate::drawishness`]); a no-op at `λ = 0`.
//!
//! # Exploration policy (so the book branches)
//!
//! Pure self-play with a deterministic engine plays the **same single line**
//! every game, so the book would be one path. To make the book a branching
//! tree/DAG we vary the opening with a **seeded ε-greedy-over-top-k** policy
//! at low plies only (design doc §4.3.7 self-play; the deliverable asks for a
//! documented seeded exploration policy):
//!
//! - At parent ply `< EXPLORE_PLIES`, score every legal move with a
//!   depth-`depth_limit` search. Collect the moves whose value is within
//!   [`EXPLORE_MARGIN`] of the best (the *near-best* set). With per-game
//!   seeded probability the game **branches**: it picks a move from the
//!   near-best set indexed by the game number, so successive games fan out
//!   over the near-best children. Otherwise it plays the search-best move.
//! - At ply `>= EXPLORE_PLIES` the game always plays the search-best move
//!   ([`decide_move`]), i.e. ordinary deterministic self-play. This bounds
//!   the book to the **opening** (book-exit condition; see below) and keeps
//!   each game cheap.
//!
//! The whole policy is driven by a single `ChaCha20Rng` seeded from
//! `cfg.seed`, so two `learn_book` runs with the same [`BookConfig`] produce
//! a **byte-identical** book (asserted by the gold serialization test).
//!
//! # Book-exit / opening-only bound
//!
//! Positions are only inserted while `parent_ply < cfg.max_book_plies`
//! (default [`DEFAULT_MAX_BOOK_PLIES`]) **and** `empties > endgame_empties`
//! (default [`DEFAULT_BOOK_ENDGAME_EMPTIES`]). The book is therefore strictly
//! an *opening* structure: a lookup never returns a move once play has left
//! the booked opening, so the book can never override the Phase-3 exact
//! endgame (which only runs at low empties anyway). [`OpeningBook::probe`]
//! additionally re-checks legality, so a stale/illegal entry is never played.

use std::collections::HashMap;
use std::io::{self, Read, Write};

use logistello_core::Zobrist;
use logistello_eval::LeafEvaluator;
use logistello_search::{EngineConfig, decide_move, terminal_score};
use othello_core::{Color, Coord, GameState, Move};

use crate::drawishness;

/// `OPB1` file magic (`"OPB1"` little-endian: `b'O' | b'P'<<8 | b'B'<<16 |
/// b'1'<<24`).
pub const BOOK_MAGIC: u32 = 0x3142_504F;

/// `OPB1` format version.
pub const BOOK_VERSION: u32 = 1;

/// Default cap on the parent ply a position is still booked at (the book is
/// an *opening* structure; design doc §4.3.7 / §5.1 uses `--depth 24` games
/// but the book itself only needs the early plies).
pub const DEFAULT_MAX_BOOK_PLIES: u32 = 20;

/// Default empties floor below which a position is **not** booked, so the
/// book can never shadow the Phase-3 exact endgame (design doc §4.5 B8
/// default is 20; the book stays well above it).
pub const DEFAULT_BOOK_ENDGAME_EMPTIES: u32 = 30;

/// Plies (from the standard start) over which the exploration policy may
/// branch the self-play game (design doc §4.3.7 self-play exploration).
pub const EXPLORE_PLIES: u32 = 8;

/// A move is "near-best" (a branch candidate) if its
/// depth-[`EXPLORE_RANK_DEPTH`] value is within this many disc-scale points
/// of the best move's value.
pub const EXPLORE_MARGIN: i32 = 4;

/// Shallow fixed depth used **only** to rank candidate moves for the
/// exploration branch (a cheap near-best probe; the *played* move at
/// non-branch plies still comes from the full depth-`depth_limit`
/// [`decide_move`]). Kept tiny so self-play stays fast — exploration only
/// needs to fan the opening out, not rank perfectly.
pub const EXPLORE_RANK_DEPTH: u32 = 2;

/// Branch denominator for the seeded ε-greedy policy: at an eligible ply a
/// single seeded `u64` is drawn; the game branches (samples the near-best
/// set) iff `draw % EXPLORE_BRANCH_DENOM == 0`, otherwise it plays the
/// search-best move. One draw per eligible ply keeps the policy
/// deterministic regardless of whether the (expensive) near-best probe runs.
pub const EXPLORE_BRANCH_DENOM: u64 = 2;

/// Per-`(position, move)` self-play aggregate (the `UpdateStatistics`
/// accumulator of design doc §4.3.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MoveStat {
    /// Sum of `final_eval` (parent side-to-move POV) over the games that
    /// played this move from this position.
    eval_sum: i64,
    /// Number of self-play games that played this move from this position.
    visits: u32,
}

impl MoveStat {
    /// Mean accumulated self-play eval (parent POV) for this move; the
    /// frontier value the negamax back-up bottoms out at when the child is
    /// not itself in the book.
    fn mean_eval(self) -> i32 {
        if self.visits == 0 {
            0
        } else {
            // Round-to-nearest; stays within the disc scale.
            let v = self.visits as i64;
            ((self.eval_sum + v / 2) / v) as i32
        }
    }
}

/// One opening-book position entry.
///
/// `best_move`, `value`, `visit_count` are the design doc §4.3.7 record
/// fields; `moves` carries the per-move self-play aggregates needed to
/// recompute the negamax back-up and the drawishness preference (the
/// `UpdateStatistics` state).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookEntry {
    /// Booked move for this position (the negamax-best, possibly shifted by
    /// drawishness). `Move::Pass` only when the position must pass.
    pub best_move: Move,
    /// Negamax-backed value of this position from its **side-to-move** POV
    /// (before back-prop = the accumulated self-play eval).
    pub value: i32,
    /// Total self-play games that visited this position.
    pub visit_count: u32,
    /// Per-move self-play aggregates (`Move` → stats), insertion-stable via
    /// the parallel `move_order`. Private: mutated only through
    /// [`OpeningBook::record`].
    moves: HashMap<MoveKey, MoveStat>,
    /// Deterministic move order (first-seen order) so iteration / tie-breaks
    /// are reproducible regardless of `HashMap` iteration order.
    move_order: Vec<Move>,
}

/// Hashable, ordered move key (Othello `Move` is `Hash + Eq`; this wrapper
/// exists only to give a stable total order for deterministic tie-breaks).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct MoveKey(Move);

impl BookEntry {
    fn new() -> Self {
        Self {
            best_move: Move::Pass,
            value: 0,
            visit_count: 0,
            moves: HashMap::new(),
            move_order: Vec::new(),
        }
    }

    /// `UpdateStatistics(move, final_eval)` (design doc §4.3.7): fold one
    /// self-play observation of `mv` with parent-POV `final_eval` into the
    /// aggregate.
    fn update(&mut self, mv: Move, final_eval: i32) {
        if !self.moves.contains_key(&MoveKey(mv)) {
            self.move_order.push(mv);
            self.moves.insert(
                MoveKey(mv),
                MoveStat {
                    eval_sum: 0,
                    visits: 0,
                },
            );
        }
        let e = self.moves.get_mut(&MoveKey(mv)).expect("just inserted");
        e.eval_sum += i64::from(final_eval);
        e.visits += 1;
        self.visit_count += 1;
    }

    /// The accumulated (pre-back-prop) self-play value of this position from
    /// its side-to-move POV: the best mean over the moves actually played
    /// here (negamax leaf when no in-book successor exists).
    fn accumulated_value(&self) -> i32 {
        let mut best = i32::MIN;
        for m in &self.move_order {
            let v = self.moves[&MoveKey(*m)].mean_eval();
            if v > best {
                best = v;
            }
        }
        if best == i32::MIN { 0 } else { best }
    }

    /// Moves recorded at this position, in first-seen (deterministic) order.
    #[must_use]
    pub fn recorded_moves(&self) -> &[Move] {
        &self.move_order
    }

    /// Mean accumulated self-play eval for `mv` (parent POV), if recorded.
    #[must_use]
    pub fn move_mean_eval(&self, mv: Move) -> Option<i32> {
        self.moves.get(&MoveKey(mv)).map(|s| s.mean_eval())
    }
}

/// The learned opening book: position → [`BookEntry`].
///
/// Keyed by the side-to-move-aware Zobrist hash, so transpositions collapse
/// to one entry and the book is a DAG. A parallel `state_of` map keeps one
/// representative [`GameState`] per key so [`negamax_backpropagate`] can
/// recompute successors without re-deriving them from a move list (and so
/// [`probe`](OpeningBook::probe) can re-verify legality).
///
/// The book owns a single [`Zobrist`] key table (built once); every internal
/// hot path hashes through it via [`Zobrist::hash`] rather than the
/// `zobrist_key` free function (which rebuilds the table on every call).
/// Both produce the **same** keys (same frozen seed), so a hand-built /
/// reloaded book is interchangeable.
#[derive(Debug, Clone)]
pub struct OpeningBook {
    entries: HashMap<u64, BookEntry>,
    /// One representative position per key (board + side to move; the
    /// fields the Zobrist key actually depends on).
    state_of: HashMap<u64, GameState>,
    /// Shared key table (built once; deterministic, frozen seed).
    zobrist: Zobrist,
}

impl Default for OpeningBook {
    fn default() -> Self {
        Self::new()
    }
}

impl OpeningBook {
    /// An empty book.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            state_of: HashMap::new(),
            zobrist: Zobrist::new(),
        }
    }

    /// This book's Zobrist key for `state` (uses the shared key table).
    #[inline]
    #[must_use]
    fn key(&self, state: &GameState) -> u64 {
        self.zobrist.hash(state)
    }

    /// Number of distinct booked positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the book has no positions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry for `state`, if booked (no legality re-check; see
    /// [`probe`](Self::probe) for the play-time lookup).
    #[must_use]
    pub fn get(&self, state: &GameState) -> Option<&BookEntry> {
        self.entries.get(&self.key(state))
    }

    /// Play-time lookup: returns the booked move for `state` **only** if it
    /// is currently legal there (and not a spurious pass while placements
    /// exist). This is the composition guard — the book never returns an
    /// illegal move and never a pass-when-moves-exist (design doc §4.3.7).
    #[must_use]
    pub fn probe(&self, state: &GameState) -> Option<Move> {
        let e = self.get(state)?;
        let legal = state.legal_moves();
        match e.best_move {
            Move::Pass => {
                if legal.is_empty() {
                    Some(Move::Pass)
                } else {
                    None
                }
            }
            m @ Move::Place(_) => {
                if legal.contains(&m) {
                    Some(m)
                } else {
                    None
                }
            }
        }
    }

    /// Folds one self-play `(state, mv, final_eval)` observation into the
    /// book (design doc §4.3.7 `UpdateStatistics`). `final_eval` is from
    /// `state`'s side-to-move POV.
    fn record(&mut self, state: &GameState, mv: Move, final_eval: i32) {
        let key = self.key(state);
        self.state_of.entry(key).or_insert_with(|| state.clone());
        let e = self.entries.entry(key).or_insert_with(BookEntry::new);
        e.update(mv, final_eval);
        // Provisional best move = current accumulated best (refined by
        // back-prop / drawishness later). Keeps a freshly recorded,
        // never-back-propagated book already usable.
        e.best_move = best_recorded_move(e);
    }

    /// Iterator over `(zobrist_key, &BookEntry)` (test/inspection helper).
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &BookEntry)> {
        self.entries.iter()
    }

    // --- OPB1 serialization (explicit little-endian, versioned) ----------

    /// Serializes the book to the explicit little-endian `OPB1` byte layout
    /// (documented in `BOOK_FORMAT.md`). Deterministic: positions are
    /// emitted in ascending `zobrist_key` order and moves in first-seen
    /// order, so equal books produce byte-identical blobs.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&BOOK_MAGIC.to_le_bytes());
        out.extend_from_slice(&BOOK_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());

        let mut keys: Vec<u64> = self.entries.keys().copied().collect();
        keys.sort_unstable();
        for key in keys {
            let e = &self.entries[&key];
            let st = &self.state_of[&key];
            out.extend_from_slice(&key.to_le_bytes());
            // Position: side to move (0=Black,1=White) + 64 cells
            // (0=empty,1=black,2=white) so loading can re-derive legality.
            out.push(match st.side_to_move {
                Color::Black => 0,
                Color::White => 1,
            });
            for row in 0..8u8 {
                for col in 0..8u8 {
                    let code = match st.board.cell(Coord::new(row, col)) {
                        None => 0u8,
                        Some(Color::Black) => 1,
                        Some(Color::White) => 2,
                    };
                    out.push(code);
                }
            }
            out.extend_from_slice(&encode_move(e.best_move).to_le_bytes());
            out.extend_from_slice(&e.value.to_le_bytes());
            out.extend_from_slice(&e.visit_count.to_le_bytes());
            out.extend_from_slice(&(e.move_order.len() as u32).to_le_bytes());
            for &m in &e.move_order {
                let s = e.moves[&MoveKey(m)];
                out.extend_from_slice(&encode_move(m).to_le_bytes());
                out.extend_from_slice(&s.eval_sum.to_le_bytes());
                out.extend_from_slice(&s.visits.to_le_bytes());
            }
        }
        out
    }

    /// Parses an `OPB1` blob produced by [`to_bytes`](Self::to_bytes).
    ///
    /// # Errors
    ///
    /// Rejects a wrong magic / version, a truncated body, trailing bytes,
    /// or an out-of-range encoded move / cell code.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut r = ByteCursor { b: bytes, p: 0 };
        let magic = r.u32()?;
        if magic != BOOK_MAGIC {
            return Err(bad(format!("bad OPB1 magic {magic:#010x}")));
        }
        let version = r.u32()?;
        if version != BOOK_VERSION {
            return Err(bad(format!("unsupported OPB1 version {version}")));
        }
        let n = r.u64()? as usize;
        let mut book = OpeningBook::new();
        let mut prev: Option<u64> = None;
        for _ in 0..n {
            let key = r.u64()?;
            if let Some(p) = prev
                && key <= p
            {
                return Err(bad("OPB1 keys not strictly ascending".into()));
            }
            prev = Some(key);
            let stm = r.u8()?;
            let side = match stm {
                0 => Color::Black,
                1 => Color::White,
                other => return Err(bad(format!("bad side byte {other}"))),
            };
            let mut st = GameState::standard_8x8();
            st.side_to_move = side;
            for row in 0..8u8 {
                for col in 0..8u8 {
                    let code = r.u8()?;
                    let c = match code {
                        0 => None,
                        1 => Some(Color::Black),
                        2 => Some(Color::White),
                        other => return Err(bad(format!("bad cell code {other}"))),
                    };
                    st.board.set(Coord::new(row, col), c);
                }
            }
            let best_move = decode_move(r.u16()?)?;
            let value = r.i32()?;
            let visit_count = r.u32()?;
            let nmoves = r.u32()? as usize;
            let mut e = BookEntry::new();
            e.best_move = best_move;
            e.value = value;
            e.visit_count = visit_count;
            for _ in 0..nmoves {
                let mv = decode_move(r.u16()?)?;
                let eval_sum = r.i64()?;
                let visits = r.u32()?;
                e.move_order.push(mv);
                e.moves.insert(MoveKey(mv), MoveStat { eval_sum, visits });
            }
            book.entries.insert(key, e);
            book.state_of.insert(key, st);
        }
        if r.p != bytes.len() {
            return Err(bad(format!(
                "trailing OPB1 bytes ({} of {})",
                bytes.len() - r.p,
                bytes.len()
            )));
        }
        Ok(book)
    }

    /// Saves the book to `path` in the `OPB1` format.
    ///
    /// # Errors
    ///
    /// Propagates filesystem errors.
    pub fn save(&self, path: &std::path::Path) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&self.to_bytes())
    }

    /// Loads an `OPB1` book from `path`.
    ///
    /// # Errors
    ///
    /// Propagates filesystem errors and [`from_bytes`](Self::from_bytes)
    /// parse errors.
    pub fn load(path: &std::path::Path) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }
}

/// The accumulated-best move of an entry (pre-back-prop): the recorded move
/// with the largest mean self-play eval; ties broken by first-seen order
/// (deterministic).
fn best_recorded_move(e: &BookEntry) -> Move {
    let mut best_m = e.move_order.first().copied().unwrap_or(Move::Pass);
    let mut best_v = i32::MIN;
    for &m in &e.move_order {
        let v = e.moves[&MoveKey(m)].mean_eval();
        if v > best_v {
            best_v = v;
            best_m = m;
        }
    }
    best_m
}

/// Encodes a [`Move`] into a `u16`: `Pass = 0xFFFF`, `Place(r,c)` = the
/// bit index `row*8+col` (`0..=63`).
fn encode_move(m: Move) -> u16 {
    match m {
        Move::Pass => 0xFFFF,
        Move::Place(c) => u16::from(c.row) * 8 + u16::from(c.col),
    }
}

/// Decodes [`encode_move`]. Rejects an out-of-range index.
fn decode_move(v: u16) -> io::Result<Move> {
    if v == 0xFFFF {
        Ok(Move::Pass)
    } else if v < 64 {
        Ok(Move::Place(Coord::new((v / 8) as u8, (v % 8) as u8)))
    } else {
        Err(bad(format!("bad encoded move {v}")))
    }
}

fn bad(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

/// Minimal little-endian byte cursor for `from_bytes`.
struct ByteCursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl ByteCursor<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        if self.p + n > self.b.len() {
            return Err(bad(format!(
                "truncated OPB1: need {n} at {} of {}",
                self.p,
                self.b.len()
            )));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> io::Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Opening-book learner configuration (design doc §4.3.7 / §5.1 / §6).
#[derive(Debug, Clone)]
pub struct BookConfig {
    /// Number of self-play games (`--num-games`; design doc §5.1: 10000).
    pub num_games: u32,
    /// Per-move search depth used during self-play (`--depth`; design doc
    /// §5.1: 24, §6 `--book-depth-values {12,18,24,30}`).
    pub depth_limit: u32,
    /// Drawishness blend `λ ∈ [0,0.5]` (`--drawishness`; design doc §5.1:
    /// 0.3, §6 range `0.0..0.5`). Clamped in [`apply_drawishness`].
    pub drawishness: f64,
    /// Master RNG seed for the exploration policy (`--seed`); two runs with
    /// the same config produce a byte-identical book.
    pub seed: u64,
    /// Cap on the parent ply still booked (opening-only bound; book-exit).
    pub max_book_plies: u32,
    /// Empties floor below which positions are not booked (keeps the book
    /// clear of the Phase-3 exact endgame).
    pub endgame_empties: u32,
}

impl Default for BookConfig {
    fn default() -> Self {
        Self {
            num_games: 1000,
            depth_limit: 6,
            drawishness: 0.3,
            seed: 1,
            max_book_plies: DEFAULT_MAX_BOOK_PLIES,
            endgame_empties: DEFAULT_BOOK_ENDGAME_EMPTIES,
        }
    }
}

/// Builds the opening book by self-play + Negamax back-prop + drawishness
/// (design doc §4.3.7 `LearnOpeningBook`).
///
/// `evaluator` is the leaf evaluator for the self-play search (Phase 3
/// `BasicEval` or Phase 4 `PatternEval`). Deterministic for a given
/// `cfg` (the exploration RNG is seeded from `cfg.seed`).
#[must_use]
pub fn learn_book<E: LeafEvaluator>(evaluator: &E, cfg: &BookConfig) -> OpeningBook {
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    let mut book = OpeningBook::new();
    let engine_cfg = EngineConfig {
        max_depth: cfg.depth_limit,
        // Self-play uses the depth-`depth_limit` heuristic search for the
        // *whole* game (design doc §4.3.7: "each move chosen by a depth-D
        // search"). We deliberately disable the Phase-3 exact-endgame
        // substitution here (`endgame_empties = 0`): it would run a full
        // ~20-ply exact solve on every late move of every self-play game
        // (minutes for thousands of games) while contributing nothing —
        // the book only records the *opening* (bounded by
        // `max_book_plies` / `endgame_empties`), and `final_eval` is the
        // game's terminal disc-difference regardless of how the endgame
        // was played. This keeps the played move a pure depth-D search,
        // exactly as the §4.3.7 pseudocode states.
        endgame_empties: 0,
        ..EngineConfig::default()
    };
    let mut rng = ChaCha20Rng::seed_from_u64(cfg.seed);

    for game in 0..cfg.num_games {
        self_play_one(evaluator, cfg, &engine_cfg, game, &mut rng, &mut book);
    }

    negamax_backpropagate(&mut book);
    apply_drawishness(&mut book, cfg.drawishness);
    book
}

/// Plays one self-play game and folds its booked `(position, move,
/// final_eval)` triples into `book` (design doc §4.3.7 `SelfPlay` +
/// `UpdateStatistics`).
fn self_play_one<E: LeafEvaluator>(
    evaluator: &E,
    cfg: &BookConfig,
    engine_cfg: &EngineConfig,
    game: u32,
    rng: &mut rand_chacha::ChaCha20Rng,
    book: &mut OpeningBook,
) {
    use logistello_search::killer::KillerTable;
    use logistello_search::tt::TranspositionTable;
    use rand::Rng;

    let z = Zobrist::new();
    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();

    let mut state = GameState::standard_8x8();
    // (position, side-to-move-POV move) pairs eligible for booking.
    let mut trace: Vec<(GameState, Move)> = Vec::new();
    let mut ply: u32 = 0;

    while !state.is_terminal() {
        let legal = state.legal_moves();
        if legal.is_empty() {
            // §4.5 B6 forced pass: same side-to-move flips, ply still
            // advances the game but a pass node is booked too (so the
            // negamax DAG includes it).
            let bookable =
                ply < cfg.max_book_plies && state.board.empty_count() > cfg.endgame_empties;
            if bookable {
                trace.push((state.clone(), Move::Pass));
            }
            state
                .apply_move(Move::Pass)
                .expect("pass legal when no placement");
            ply += 1;
            continue;
        }

        // Depth-`depth_limit` search picks the move (design doc §4.3.7
        // "each move chosen by a depth-D search").
        let res = decide_move(&state, evaluator, engine_cfg, &mut tt, &mut killers, &z);
        let mut chosen = res.best_move;

        // Seeded ε-greedy exploration at low plies so the book branches.
        // Exactly one seeded draw per eligible ply (deterministic
        // regardless of whether the near-best probe runs); the expensive
        // probe only runs on the (rarer) branch plies.
        if ply < EXPLORE_PLIES {
            let r: u64 = rng.r#gen();
            let branch = (r ^ u64::from(game).wrapping_mul(0x9E37_79B9))
                .is_multiple_of(EXPLORE_BRANCH_DENOM);
            if branch {
                let near = near_best_moves(&state, evaluator, engine_cfg, &z);
                if near.len() > 1 {
                    let idx = (r >> 1) as usize % near.len();
                    chosen = near[idx];
                }
            }
        }

        let bookable = ply < cfg.max_book_plies && state.board.empty_count() > cfg.endgame_empties;
        if bookable {
            trace.push((state.clone(), chosen));
        }
        state.apply_move(chosen).expect("chosen move is legal");
        ply += 1;
    }

    // final_eval = the game's terminal disc-difference, projected to each
    // booked position's side-to-move POV (design doc §4.3.7 final_eval; B6
    // terminal_score sign convention reused via Black-relative diff).
    let black_diff =
        state.board.count(Color::Black) as i32 - state.board.count(Color::White) as i32;
    for (pos, mv) in &trace {
        let final_eval = match pos.side_to_move {
            Color::Black => black_diff,
            Color::White => -black_diff,
        };
        book.record(pos, *mv, final_eval);
    }
}

/// The near-best move set at `state`: every legal move whose own
/// depth-[`EXPLORE_RANK_DEPTH`] value is within [`EXPLORE_MARGIN`] of the
/// best.
///
/// Each move is scored by a **cheap shallow** fixed-depth NegaScout of the
/// child (negated to parent POV). This only ranks branch candidates so the
/// book fans out; the move actually *played* at non-branch plies still
/// comes from the full depth-`depth_limit` [`decide_move`]. Order is the
/// legal-move generation order (deterministic). A fresh TT/killer per child
/// keeps the ranking independent of search history (reproducible).
fn near_best_moves<E: LeafEvaluator>(
    state: &GameState,
    evaluator: &E,
    engine_cfg: &EngineConfig,
    z: &Zobrist,
) -> Vec<Move> {
    use logistello_search::killer::KillerTable;
    use logistello_search::tt::TranspositionTable;

    let legal = state.legal_moves();
    let rank_depth = EXPLORE_RANK_DEPTH.min(engine_cfg.max_depth);
    let mut scored: Vec<(Move, i32)> = Vec::with_capacity(legal.len());
    for &m in &legal {
        let mut child = state.clone();
        child.apply_move(m).expect("legal move applies");
        let v = if child.is_terminal() {
            terminal_score(&child)
        } else {
            let mut tt = TranspositionTable::new();
            let mut killers = KillerTable::new();
            logistello_search::search_with(
                &child,
                rank_depth,
                evaluator,
                &mut tt,
                &mut killers,
                z,
                logistello_search::SearchConfig::default(),
            )
            .value
        };
        scored.push((m, -v)); // parent POV
    }
    let best = scored.iter().map(|&(_, v)| v).max().unwrap_or(0);
    scored
        .into_iter()
        .filter(|&(_, v)| best - v <= EXPLORE_MARGIN)
        .map(|(m, _)| m)
        .collect()
}

/// Recomputes every in-book position's value as the **negamax** of its
/// in-book successors (design doc §4.3.7 `NegamaxBackpropagate`).
///
/// For a position `p`:
///
/// - terminal `p` (shouldn't be booked, but handled): `value =
///   terminal_score(p)`, `best_move = Pass`.
/// - must-pass `p` (no placement): the single successor is `pass(p)`; its
///   value is negated with **no** depth / empties change (design doc §4.5
///   B6). If `pass(p)` is in the book, use its (back-propagated) value;
///   otherwise the accumulated self-play value.
/// - otherwise: for each **recorded** move `m`, the child `c = p.m`. If `c`
///   is in the book use `-c.value`; else the frontier value falls back to
///   `0`/the accumulated estimate. `value = max_m (those)`, `best_move`
///   attains it (first recorded order on ties — deterministic).
///
/// Convergence: a non-pass move strictly decreases `empties`, and a pass
/// chains at most a bounded number of times before terminal, so the
/// transposition DAG is acyclic in `(empties, pass-rank)`. We process
/// positions by **ascending empties** (successors finalised before
/// predecessors) and iterate the whole sweep to a fixpoint (handles the
/// bounded equal-empties pass layer and any transposition ordering
/// effects). Guaranteed to terminate; in practice 2 sweeps suffice.
pub fn negamax_backpropagate(book: &mut OpeningBook) {
    // Snapshot keys ordered by ascending empties so children (fewer
    // empties) are processed before parents.
    let mut order: Vec<(u32, u64)> = book
        .state_of
        .iter()
        .map(|(&k, s)| (s.board.empty_count(), k))
        .collect();
    // Ascending empties; within equal empties, stable by key for
    // determinism.
    order.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let max_sweeps = order.len() + 2;
    for _ in 0..max_sweeps {
        let mut changed = false;
        for &(_, key) in &order {
            let state = book.state_of[&key].clone();
            let (new_val, new_best) = backed_value(book, &state);
            let e = book.entries.get_mut(&key).expect("keyed entry exists");
            if e.value != new_val || e.best_move != new_best {
                e.value = new_val;
                e.best_move = new_best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

/// The negamax-backed `(value, best_move)` of `state` given the current
/// book values (one relaxation step of [`negamax_backpropagate`]).
fn backed_value(book: &OpeningBook, state: &GameState) -> (i32, Move) {
    if state.is_terminal() {
        return (terminal_score(state), Move::Pass);
    }
    let legal = state.legal_moves();
    let key = book.key(state);
    let entry = book.entries.get(&key);

    if legal.is_empty() {
        // Must pass: value = -value(pass(state)), no depth/empties change.
        let mut passed = state.clone();
        passed
            .apply_move(Move::Pass)
            .expect("pass legal when no placement");
        let child_v = child_value(book, &passed);
        return (-child_v, Move::Pass);
    }

    // Score every recorded move via its child; fall back to per-move mean
    // when the child is not booked (frontier).
    let mut best_v = i32::MIN;
    let mut best_m = legal[0];
    let recorded: Vec<Move> = entry
        .map(|e| e.recorded_moves().to_vec())
        .unwrap_or_default();
    for m in recorded {
        // Only placements lead to a distinct child here (pass handled
        // above); a recorded pass at a movable node is ignored.
        let Move::Place(_) = m else { continue };
        if !legal.contains(&m) {
            continue;
        }
        let mut child = state.clone();
        child.apply_move(m).expect("recorded legal move applies");
        let cv = child_value(book, &child);
        let v = -cv;
        if v > best_v {
            best_v = v;
            best_m = m;
        }
    }
    if best_v == i32::MIN {
        // No usable recorded successor: fall back to the accumulated value.
        let v = entry.map(BookEntry::accumulated_value).unwrap_or(0);
        return (v, best_m);
    }
    (best_v, best_m)
}

/// Value of a child position from **its own** side-to-move POV: the
/// back-propagated book value if booked, the exact terminal score if
/// terminal, else `0` (an unseen frontier child contributes nothing).
fn child_value(book: &OpeningBook, child: &GameState) -> i32 {
    if child.is_terminal() {
        return terminal_score(child);
    }
    let key = book.key(child);
    if let Some(e) = book.entries.get(&key) {
        e.value
    } else {
        0
    }
}

/// Shifts each position's `best_move` toward children with higher
/// opponent-error potential (design doc §4.3.7 `ApplyDrawishness`,
/// Buro 1999) via [`crate::drawishness`].
///
/// For every booked, movable position `p` with at least one recorded
/// successor, recompute the preference
/// `score(m, λ) = (1-λ)·negamax_value(child) + λ·trap(child)` and set
/// `best_move` to the maximiser (ties → higher pure negamax value, then
/// first recorded order).
///
/// **`λ = 0` is an exact no-op**: `score = negamax_value`, so `best_move`
/// stays the [`negamax_backpropagate`] best (asserted by the gold test).
/// `lambda` is clamped to `[0, 0.5]` (design doc §6).
pub fn apply_drawishness(book: &mut OpeningBook, lambda: f64) {
    let l = drawishness::clamp_lambda(lambda);

    let keys: Vec<u64> = book.entries.keys().copied().collect();
    for key in keys {
        let state = book.state_of[&key].clone();
        if state.is_terminal() {
            continue;
        }
        let legal = state.legal_moves();
        if legal.is_empty() {
            continue; // must-pass: single forced successor, nothing to shift
        }
        let recorded: Vec<Move> = book
            .entries
            .get(&key)
            .map(|e| e.recorded_moves().to_vec())
            .unwrap_or_default();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_m: Option<Move> = None;
        let mut best_v = i32::MIN;
        for m in recorded {
            let Move::Place(_) = m else { continue };
            if !legal.contains(&m) {
                continue;
            }
            let mut child = state.clone();
            child.apply_move(m).expect("recorded legal move applies");
            // Parent-POV negamax value of the child.
            let nv = -child_value(book, &child);
            // trap = spread of the OPPONENT's in-book replies at `child`.
            let trap = trap_of_child(book, &child);
            let s = drawishness::score(nv, trap, l);
            // Strict improvement only; equal score keeps the higher pure
            // negamax value, then first recorded order. At λ=0
            // `s == nv as f64`, so this reproduces the negamax best exactly.
            if s > best_score || (s == best_score && nv > best_v) {
                best_score = s;
                best_v = nv;
                best_m = Some(m);
            }
        }
        if let Some(m) = best_m {
            book.entries
                .get_mut(&key)
                .expect("keyed entry exists")
                .best_move = m;
        }
    }
}

/// The drawishness `trap` of a child position: the spread of the
/// **opponent's** in-book continuation values at `child` (opponent POV).
/// `child.side_to_move` is already the opponent, so the child entry's
/// per-recorded-move parent-POV negamax values are exactly the opponent's
/// reply values. Returns `0` when `child` is not booked or has < 2 replies.
fn trap_of_child(book: &OpeningBook, child: &GameState) -> i32 {
    if child.is_terminal() {
        return 0;
    }
    let key = book.key(child);
    let Some(e) = book.entries.get(&key) else {
        return 0;
    };
    let legal = child.legal_moves();
    if legal.is_empty() {
        return 0;
    }
    let mut vals: Vec<i32> = Vec::new();
    for &m in e.recorded_moves() {
        let Move::Place(_) = m else { continue };
        if !legal.contains(&m) {
            continue;
        }
        let mut gc = child.clone();
        gc.apply_move(m).expect("recorded legal move applies");
        vals.push(-child_value(book, &gc)); // opponent (child STM) POV
    }
    drawishness::trap(&vals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::BasicEval;

    #[test]
    fn move_codec_roundtrips() {
        for r in 0..8u8 {
            for c in 0..8u8 {
                let m = Move::Place(Coord::new(r, c));
                assert_eq!(decode_move(encode_move(m)).unwrap(), m);
            }
        }
        assert_eq!(decode_move(encode_move(Move::Pass)).unwrap(), Move::Pass);
        assert!(decode_move(64).is_err());
        assert!(decode_move(0xFFFE).is_err());
    }

    #[test]
    fn learn_book_is_deterministic_and_nonempty() {
        let cfg = BookConfig {
            num_games: 12,
            depth_limit: 3,
            drawishness: 0.3,
            seed: 7,
            ..BookConfig::default()
        };
        let a = learn_book(&BasicEval::default(), &cfg);
        let b = learn_book(&BasicEval::default(), &cfg);
        assert!(!a.is_empty());
        assert_eq!(a.to_bytes(), b.to_bytes(), "same cfg => identical book");
        // Covers the initial position.
        assert!(a.get(&GameState::standard_8x8()).is_some());
    }

    #[test]
    fn book_grows_with_more_games() {
        let base = BookConfig {
            num_games: 4,
            depth_limit: 3,
            seed: 3,
            ..BookConfig::default()
        };
        let more = BookConfig {
            num_games: 40,
            ..base.clone()
        };
        let small = learn_book(&BasicEval::default(), &base);
        let big = learn_book(&BasicEval::default(), &more);
        assert!(
            big.len() >= small.len(),
            "more games must not shrink the book ({} vs {})",
            big.len(),
            small.len()
        );
        assert!(big.len() > 1);
    }
}
