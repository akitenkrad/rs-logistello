//! Search driver + engine player (design doc §4.3.3 / §4.5 B8).
//!
//! [`decide_move`] is the single decision entry point. It implements Buro's
//! three-phase protocol switch (design doc §4.5 B8):
//!
//! - **endgame** (empties `<= endgame_empties`, default 20 = the Logistello
//!   1997 Murakami-engine value): run the cheap WLD probe (for
//!   logging / staging) and then the exact disc-difference solver; return
//!   the exact value and best move with `depth = empties`.
//! - **selective midgame** (more empties than the threshold): the Phase 2
//!   iterative-deepening NegaScout with the supplied heuristic evaluator.
//!
//! [`LogistelloPlayer`] wraps [`decide_move`] behind the reused
//! `othello_player::Player` trait so the engine can play full games via
//! `othello_engine::GameEngine`. It owns its TT / killer table / Zobrist /
//! evaluator / config.

use logistello_core::Zobrist;
use logistello_eval::{BasicEval, LeafEvaluator, PatternEval};
use othello_core::{Color, GameState, Move};
use othello_player::{Player, PlayerError};

use crate::alphabeta::SearchConfig;
use crate::endgame::{best_endgame_move, solve_wld};
use crate::iterative::{SearchResult, search_with};
use crate::killer::KillerTable;
use crate::multi_probcut::MultiProbCutConfig;
use crate::probcut::ProbCutConfig;
use crate::tt::TranspositionTable;

/// Default empty-square threshold below which exact endgame play kicks in
/// (design doc §4.5 B8: Logistello 1997 / Murakami-engine canonical value;
/// Edax-parity is 10, exposed as a parameter).
pub const DEFAULT_ENDGAME_EMPTIES: u32 = 20;

/// Default iterative-deepening max depth for the selective midgame.
pub const DEFAULT_MAX_DEPTH: u32 = 8;

/// Engine configuration (design doc §4.5 B8 / §6 sensitivity target).
///
/// [`Clone`] but not `Copy` because [`MultiProbCutConfig`] owns a per-cell
/// coefficient map (the OFF default keeps it empty, so a default clone is
/// cheap).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Selective-midgame iterative-deepening max depth (plies).
    pub max_depth: u32,
    /// Switch to exact endgame play once `empties <= endgame_empties`
    /// (design doc §4.5 B8; default 20, Edax-parity 10).
    pub endgame_empties: u32,
    /// Use the transposition table.
    pub use_tt: bool,
    /// Use killer-move ordering.
    pub use_killers: bool,
    /// Single-ProbCut configuration (design doc §4.3.4 / §4.5 B7).
    /// [`ProbCutConfig::default`] is **disabled** so the default engine is
    /// exactly the Phase 3 selective-midgame search.
    pub probcut: ProbCutConfig,
    /// Multi-ProbCut cascade configuration (design doc §4.3.5 / §4.5 B7).
    /// [`MultiProbCutConfig::default`] is **disabled** so the default engine
    /// is exactly the Phase 3 selective-midgame search; when enabled it
    /// supersedes single ProbCut at the cascade heights (`3..=13`).
    pub multi_probcut: MultiProbCutConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            endgame_empties: DEFAULT_ENDGAME_EMPTIES,
            use_tt: true,
            use_killers: true,
            probcut: ProbCutConfig::default(),
            multi_probcut: MultiProbCutConfig::default(),
        }
    }
}

impl EngineConfig {
    /// The Phase 2 [`SearchConfig`] implied by this engine config.
    #[must_use]
    fn search_config(&self) -> SearchConfig {
        SearchConfig {
            use_tt: self.use_tt,
            use_killers: self.use_killers,
            // ProbCut + Multi-ProbCut wired through the engine config
            // (design doc §4.5 B7). Both default OFF.
            probcut: self.probcut,
            multi_probcut: self.multi_probcut.clone(),
        }
    }
}

/// Which protocol phase [`decide_move`] used for a position. Exposed so
/// tests (and callers/logging) can observe the routing decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatch {
    /// Exact endgame: WLD probe + exact disc-difference solve.
    Endgame,
    /// Selective midgame: Phase 2 iterative-deepening NegaScout.
    Midgame,
}

/// Decides the move for `root` under Buro's three-phase protocol
/// (design doc §4.5 B8), also returning the routing decision so callers /
/// tests can observe which phase ran.
///
/// - `empties <= cfg.endgame_empties` -> endgame. Runs [`solve_wld`] first
///   (cheap WLD, useful for logging / staging — its value is intentionally
///   discarded here) then [`best_endgame_move`] for the exact value and
///   move. The result's `depth` is the empty count (the exact search
///   reaches the terminal, design doc §4.5 B8).
/// - otherwise -> selective midgame: Phase 2 [`search_with`] to
///   `cfg.max_depth` with `eval`.
pub fn decide_move_with_dispatch<E: LeafEvaluator>(
    root: &GameState,
    eval: &E,
    cfg: &EngineConfig,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &Zobrist,
) -> (Dispatch, SearchResult) {
    let empties = root.board.empty_count();

    if empties <= cfg.endgame_empties {
        // Phase 2 (WLD): cheap sign-only probe. Computed for logging /
        // staging parity with Buro's protocol; the exact solve below
        // supersedes it for the actual decision.
        let _wld = solve_wld(root, tt, killers, zobrist);
        // Phase 3 (exact disc-difference maximisation).
        let (best_move, value) = best_endgame_move(root, tt, killers, zobrist);
        let result = SearchResult {
            best_move,
            value,
            depth: empties,
            nodes: 0,
        };
        return (Dispatch::Endgame, result);
    }

    // Selective midgame: Phase 2 iterative deepening.
    let result = search_with(
        root,
        cfg.max_depth,
        eval,
        tt,
        killers,
        zobrist,
        cfg.search_config(),
    );
    (Dispatch::Midgame, result)
}

/// Decides the move for `root` under Buro's three-phase protocol
/// (design doc §4.5 B8) — the spec entry point returning just the
/// [`SearchResult`]. See [`decide_move_with_dispatch`] for the routing
/// decision.
pub fn decide_move<E: LeafEvaluator>(
    root: &GameState,
    eval: &E,
    cfg: &EngineConfig,
    tt: &mut TranspositionTable,
    killers: &mut KillerTable,
    zobrist: &Zobrist,
) -> SearchResult {
    decide_move_with_dispatch(root, eval, cfg, tt, killers, zobrist).1
}

/// Midgame leaf evaluator a [`LogistelloPlayer`] can carry: the Phase 3
/// [`BasicEval`] (disc + mobility) or the Phase 4 [`PatternEval`]
/// (Edax-準拠 learned pattern model, design doc §4.3.3 / §4.4 B1-B4).
///
/// Both arms satisfy the [`LeafEvaluator`] contract (side-to-move POV,
/// disc-scale, pure), so the search's minimax invariants hold for either.
#[derive(Debug, Clone)]
pub enum EngineEval {
    /// Phase 3 disc-count + mobility evaluator.
    Basic(BasicEval),
    /// Phase 4 learned pattern/regression evaluator (boxed: it owns the
    /// large `EvalWeights` table — keeps the enum small).
    Pattern(Box<PatternEval>),
}

impl LeafEvaluator for EngineEval {
    #[inline]
    fn eval(&self, state: &GameState) -> i32 {
        match self {
            EngineEval::Basic(e) => e.eval(state),
            EngineEval::Pattern(e) => e.eval(state),
        }
    }
}

/// Logistello engine as an `othello_player::Player`.
///
/// Owns its transposition table, killer table, Zobrist hasher, evaluator,
/// and config; every `select_move` runs [`decide_move`]. The TT / killers
/// persist across moves of a game (cleared on [`Player::reset`]) so deeper
/// re-searches of revisited positions are cheaper while never changing a
/// value (Phase 2 invariants).
pub struct LogistelloPlayer {
    name: String,
    color: Color,
    config: EngineConfig,
    eval: EngineEval,
    tt: TranspositionTable,
    killers: KillerTable,
    zobrist: Zobrist,
}

impl LogistelloPlayer {
    /// Builds a player for `color` with the given engine config, using
    /// [`BasicEval::default`] as the midgame leaf evaluator.
    #[must_use]
    pub fn new(color: Color, config: EngineConfig) -> Self {
        Self {
            name: "Logistello".to_string(),
            color,
            config,
            eval: EngineEval::Basic(BasicEval::default()),
            tt: TranspositionTable::new(),
            killers: KillerTable::new(),
            zobrist: Zobrist::new(),
        }
    }

    /// Builds a player for `color` that uses the Phase 4 [`PatternEval`]
    /// (learned `LGW1` weights) as the midgame leaf evaluator instead of
    /// [`BasicEval`].
    #[must_use]
    pub fn with_pattern(color: Color, config: EngineConfig, pattern: PatternEval) -> Self {
        Self {
            name: "Logistello".to_string(),
            color,
            config,
            eval: EngineEval::Pattern(Box::new(pattern)),
            tt: TranspositionTable::new(),
            killers: KillerTable::new(),
            zobrist: Zobrist::new(),
        }
    }

    /// The engine config in use.
    #[must_use]
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Runs the decision pipeline on `state`, returning the routing
    /// decision and the full search result (exposed for tests / tooling).
    pub fn decide(&mut self, state: &GameState) -> (Dispatch, SearchResult) {
        decide_move_with_dispatch(
            state,
            &self.eval,
            &self.config,
            &mut self.tt,
            &mut self.killers,
            &self.zobrist,
        )
    }
}

impl Player for LogistelloPlayer {
    fn name(&self) -> &str {
        &self.name
    }

    fn color(&self) -> Color {
        self.color
    }

    fn select_move(&mut self, state: &GameState) -> Result<Move, PlayerError> {
        let (_dispatch, result) = self.decide(state);
        // `GameEngine` already substitutes `Move::Pass` when the side has no
        // legal move, so `select_move` is only called on movable positions;
        // returning the engine's chosen placement is correct. We still map
        // a `Pass` faithfully for direct callers / must-pass nodes.
        Ok(result.best_move)
    }

    fn reset(&mut self) {
        self.tt = TranspositionTable::new();
        self.killers = KillerTable::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logistello_eval::DiscDiffEval;

    fn fresh() -> (TranspositionTable, KillerTable, Zobrist) {
        (
            TranspositionTable::new(),
            KillerTable::new(),
            Zobrist::new(),
        )
    }

    #[test]
    fn start_position_routes_to_midgame() {
        let s = GameState::standard_8x8(); // 60 empties >> 20
        let cfg = EngineConfig::default();
        let (mut tt, mut k, z) = fresh();
        let (d, r) =
            decide_move_with_dispatch(&s, &BasicEval::default(), &cfg, &mut tt, &mut k, &z);
        assert_eq!(d, Dispatch::Midgame);
        assert_eq!(r.depth, cfg.max_depth);
        assert!(s.legal_moves().contains(&r.best_move));
    }

    #[test]
    fn near_terminal_routes_to_endgame() {
        // Deterministic walk down to <= endgame threshold (use a small
        // threshold so the exact solve is cheap).
        let cfg = EngineConfig {
            max_depth: 4,
            endgame_empties: 12,
            ..EngineConfig::default()
        };
        let mut s = GameState::standard_8x8();
        while s.board.empty_count() > cfg.endgame_empties && !s.is_terminal() {
            let mv = s.legal_moves();
            if mv.is_empty() {
                s.apply_move(Move::Pass).unwrap();
            } else {
                s.apply_move(mv[0]).unwrap();
            }
        }
        let (mut tt, mut k, z) = fresh();
        let (d, r) = decide_move_with_dispatch(&s, &DiscDiffEval, &cfg, &mut tt, &mut k, &z);
        assert_eq!(d, Dispatch::Endgame);
        assert_eq!(r.depth, s.board.empty_count());
        assert_eq!(r.value % 2, 0, "exact endgame value is even");
    }

    #[test]
    fn player_select_move_is_legal_and_deterministic() {
        let s = GameState::standard_8x8();
        let cfg = EngineConfig {
            max_depth: 4,
            ..EngineConfig::default()
        };
        let mut p1 = LogistelloPlayer::new(Color::Black, cfg.clone());
        let mut p2 = LogistelloPlayer::new(Color::Black, cfg);
        let m1 = p1.select_move(&s).unwrap();
        let m2 = p2.select_move(&s).unwrap();
        assert_eq!(m1, m2, "engine is deterministic");
        assert!(s.legal_moves().contains(&m1));
    }
}
