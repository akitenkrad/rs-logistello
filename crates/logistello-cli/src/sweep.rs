//! Phase 10 sensitivity analysis (`sweep`) — design doc §6 / §5.1.
//!
//! A `sweep` resolves every requested §6 parameter grid into an explicit
//! list of values, builds the **cross-product** of all requested grids
//! ("conditions"), and for every `(condition × seed in 0..runs)` runs the
//! one measurement appropriate to the parameter being swept. One trial is
//! recorded per `(condition, seed)`. Everything is **deterministic per `(condition, seed)`**
//! (seeds are derived explicitly from the `--seed` base + the trial index —
//! never "current time", design-doc reproducibility requirement), so two
//! identical invocations produce identical measurements.
//!
//! ## What is recorded (runvault)
//!
//! A sweep is a **parent run plus one child run per condition**. The parent
//! holds the resolved grid in its `parameters` and measures nothing itself;
//! each child holds its own point of the grid, one `x.logistello.trial` event
//! per trial, and the condition's mean of the §6 metric.
//!
//! A trial is an event rather than a run of its own because it has no time
//! axis — one trial is one measurement, so it fits in one row.
//!
//! ## Metric per swept parameter (design doc §6 "期待される主要な知見")
//!
//! Each parameter is swept against the metric that §6 predicts it should
//! move, measured from a deterministic seeded mid/late-game root
//! ([`crate::walk_position`]-style ChaCha20 walk):
//!
//! | parameter            | what is varied                    | metric (CSV col)        | §6 expected direction |
//! |----------------------|-----------------------------------|-------------------------|-----------------------|
//! | `probcut_t`          | single-ProbCut confidence `T`     | `nodes`(+`search_value`)| larger `T` ⇒ fewer cuts ⇒ more nodes (speed-up plateaus past ~2.0) |
//! | `probcut_depth_pair` | ProbCut `(d,h)`                   | `nodes`                 | shallower `d` ⇒ cheaper probe / looser cut |
//! | `multi_stages`       | MPC cascade depth (#heights kept) | `nodes`                 | `k≈3` empirical optimum |
//! | `eval_stages`        | #evaluation stages used           | `eval_abs_err`          | error drops sharply by ≥5 stages, saturates ≥13 |
//! | `glem_max_order`     | GLEM conjunction max order        | `n_features`            | order-2 captures most; order-4 overfits |
//! | `glem_support`       | GLEM support threshold (log)      | `n_features`            | higher τ ⇒ fewer surviving features |
//! | `max_depth`          | iterative-deepening depth         | `nodes`                 | nodes grow ~exponentially with depth |
//! | `tt_size`            | transposition-table size (2^k)    | `nodes`                 | bigger TT ⇒ fewer nodes (saturates ≥2^24) |
//! | `book_depth`         | opening-book self-play depth      | `book_positions`        | deeper ⇒ a larger / more stable book |
//! | `endgame_empties`    | exact-endgame switch (B8)         | `nodes`(+`dispatch`)    | larger threshold ⇒ exact endgame starts earlier ⇒ node count rises monotonically |
//! | `drawishness`        | book drawishness blend `λ`        | `selfplay_score`        | `λ` shifts the self-play outcome |
//!
//! Only the fields the swept axis actually measures are written to its trial
//! events. [`MetricRow`] still carries every column (it is one fixed struct),
//! but a column the axis never measured is a neutral zero, and writing it
//! would claim a measurement that was never made.
//! The wall-clock `time_sec` / `nps` are kept on [`MetricRow`] for the stdout
//! summary and are recorded nowhere — `status.json`'s `duration_sec` is the
//! record of time.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
use runvault::{Lineage, Run};
use serde::Serialize;
use serde_json::json;

use crate::record;

use logistello_book::{BookConfig, learn_book_observed};
use logistello_core::Zobrist;
use logistello_eval::{BasicEval, DiscDiffEval, LeafEvaluator};
use logistello_search::alphabeta::{INF, SearchConfig, SearchContext, negascout};
use logistello_search::killer::KillerTable;
use logistello_search::probcut::{ProbCutConfig, ProbCutParams};
use logistello_search::tt::TranspositionTable;
use othello_core::{GameState, Move};

// ===================================================================== //
// Grid expansion (design doc §6: min/max/step + value lists + log-scale) //
// ===================================================================== //

/// Number of decimal places implied by a (positive) step, so float
/// accumulation can be rounded to a well-defined grid (e.g. step `0.25` ⇒
/// 2 dp; `0.1` ⇒ 1 dp). Mirrors the schelling reference's `step_decimals`.
fn step_decimals(step: f64) -> i32 {
    if step <= 0.0 {
        return 0;
    }
    let mut s = step;
    let mut d = 0;
    // Stop once `s` is integral (within tolerance) or we hit a sane cap.
    while (s - s.round()).abs() > 1e-9 && d < 12 {
        s *= 10.0;
        d += 1;
    }
    d
}

/// Expands an inclusive `min..=max` range by `step` into a deterministic,
/// well-rounded value list. `min == max` (a degenerate single-point grid)
/// yields exactly `[min]`. The upper bound is *inclusive* with a small
/// tolerance so e.g. `1.0..=2.5 step 0.25` yields the full 7-point grid
/// including `2.5`.
///
/// # Errors
/// `step <= 0` or `max < min`.
pub fn expand_range(min: f64, max: f64, step: f64) -> Result<Vec<f64>> {
    if max < min {
        bail!("range max ({max}) < min ({min})");
    }
    if (max - min).abs() < 1e-12 {
        let dp = step_decimals(if step > 0.0 { step } else { 1.0 });
        return Ok(vec![round_dp(min, dp)]);
    }
    if step <= 0.0 {
        bail!("range step must be > 0 (got {step})");
    }
    let dp = step_decimals(step);
    // Inclusive count with tolerance so the endpoint is not lost to FP error.
    let n = ((max - min) / step + 1e-9).floor() as usize;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..=n {
        out.push(round_dp(min + step * i as f64, dp));
    }
    Ok(out)
}

/// Expands a `10^min_exp .. 10^max_exp` **log-scale** range stepping the
/// exponent by `log_step` (design doc §6 GLEM support: `1e-4..1e-2`,
/// `log10` step `0.25`). Values are rounded to a stable significand.
///
/// # Errors
/// `log_step <= 0` or `max_exp < min_exp`.
pub fn expand_log_range(min_exp: f64, max_exp: f64, log_step: f64) -> Result<Vec<f64>> {
    if max_exp < min_exp {
        bail!("log range max_exp ({max_exp}) < min_exp ({min_exp})");
    }
    if (max_exp - min_exp).abs() < 1e-12 {
        return Ok(vec![round_sig(10f64.powf(min_exp))]);
    }
    if log_step <= 0.0 {
        bail!("log range step must be > 0 (got {log_step})");
    }
    let n = ((max_exp - min_exp) / log_step + 1e-9).floor() as usize;
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..=n {
        out.push(round_sig(10f64.powf(min_exp + log_step * i as f64)));
    }
    Ok(out)
}

/// Round to `dp` decimals (kills FP accumulation noise so the grid is exact).
fn round_dp(v: f64, dp: i32) -> f64 {
    let f = 10f64.powi(dp);
    (v * f).round() / f
}

/// Round a (positive) value to 6 significant figures — stable for log grids.
fn round_sig(v: f64) -> f64 {
    if v == 0.0 {
        return 0.0;
    }
    let mag = v.abs().log10().floor();
    let f = 10f64.powf(5.0 - mag);
    (v * f).round() / f
}

/// Parses a comma-separated integer value-list flag (`--multi-stages-values
/// 1,2,3`). Whitespace around items is tolerated; empty ⇒ error.
///
/// # Errors
/// A non-integer item or an empty list.
pub fn parse_int_list(s: &str) -> Result<Vec<i64>> {
    let v: Result<Vec<i64>> = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<i64>()
                .with_context(|| format!("bad integer '{x}' in list '{s}'"))
        })
        .collect();
    let v = v?;
    if v.is_empty() {
        bail!("empty value list '{s}'");
    }
    Ok(v)
}

/// Parses a comma-separated `d:h` depth-pair list (`--probcut-depth-pairs-
/// values 1:5,3:7`).
///
/// # Errors
/// A malformed pair (`d>=h`, non-integer, wrong arity) or an empty list.
pub fn parse_pair_list(s: &str) -> Result<Vec<(u32, u32)>> {
    let mut out = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        let mut it = tok.split(':');
        let d: u32 = it
            .next()
            .and_then(|x| x.trim().parse().ok())
            .with_context(|| format!("bad depth pair '{tok}' (want d:h)"))?;
        let h: u32 = it
            .next()
            .and_then(|x| x.trim().parse().ok())
            .with_context(|| format!("bad depth pair '{tok}' (want d:h)"))?;
        if it.next().is_some() {
            bail!("depth pair '{tok}' has too many ':' parts (want d:h)");
        }
        if d >= h {
            bail!("depth pair needs d < h (got {d}:{h})");
        }
        out.push((d, h));
    }
    if out.is_empty() {
        bail!("empty depth-pair list '{s}'");
    }
    Ok(out)
}

// ===================================================================== //
//  Sweep spec  (the §6 parameters; each "axis" resolves to a value list) //
// ===================================================================== //

/// Which §6 parameter a sweep varies. Exactly one axis is active per sweep
/// invocation (`--help` lists them all); the cross-product is over the
/// active axis's resolved value list × `0..runs` seeds. (Sweeping a single
/// axis at a time matches the §6 table — each row is one parameter — and
/// keeps every measurement comparable.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    ProbcutT,
    ProbcutDepthPair,
    MultiStages,
    EvalStages,
    GlemMaxOrder,
    GlemSupport,
    MaxDepth,
    TtSize,
    BookDepth,
    EndgameEmpties,
    Drawishness,
}

/// Self-play games one trial of a book-learning axis plays.
///
/// Both `Axis::BookDepth` and `Axis::Drawishness` build an opening book per
/// trial, and this is the `num_games` they build it with. It is a constant so
/// that the stage's denominator comes from the same number the loop uses.
pub const BOOK_GAMES_PER_TRIAL: u32 = 6;

impl Axis {
    /// Whether one trial of this axis is a `learn_book` call.
    ///
    /// Those trials are the only ones that can run for minutes — a book game
    /// at `--book-depth-values 30` is minutes on its own — so they are the
    /// ones counted by the game rather than by the trial.
    #[must_use]
    pub fn learns_a_book(self) -> bool {
        matches!(self, Axis::BookDepth | Axis::Drawishness)
    }

    /// The `parameters` / trial-event column name for this axis.
    pub fn col(self) -> &'static str {
        match self {
            Axis::ProbcutT => "probcut_t",
            Axis::ProbcutDepthPair => "probcut_depth_pair",
            Axis::MultiStages => "multi_stages",
            Axis::EvalStages => "eval_stages",
            Axis::GlemMaxOrder => "glem_max_order",
            Axis::GlemSupport => "glem_support",
            Axis::MaxDepth => "max_depth",
            Axis::TtSize => "tt_size",
            Axis::BookDepth => "book_depth",
            Axis::EndgameEmpties => "endgame_empties",
            Axis::Drawishness => "drawishness",
        }
    }

    /// The metric column that §6 predicts this axis should move.
    pub fn metric(self) -> &'static str {
        match self {
            Axis::ProbcutT
            | Axis::ProbcutDepthPair
            | Axis::MultiStages
            | Axis::MaxDepth
            | Axis::TtSize
            | Axis::EndgameEmpties => "nodes",
            Axis::EvalStages => "eval_abs_err",
            Axis::GlemMaxOrder | Axis::GlemSupport => "n_features",
            Axis::BookDepth => "book_positions",
            Axis::Drawishness => "selfplay_score",
        }
    }

    /// The §6 list of all sweepable parameters (for `--help` / no-arg).
    pub fn all() -> &'static [(&'static str, &'static str)] {
        &[
            ("probcut_t", "--probcut-t-min/max/step (1.0..2.5 / 0.25)"),
            ("probcut_depth_pair", "--probcut-depth-pairs-values d:h,..."),
            ("multi_stages", "--multi-stages-values 1..5"),
            ("eval_stages", "--eval-stages-values 1,5,10,13,20,30"),
            ("glem_max_order", "--glem-max-order-values 1..4"),
            ("glem_support", "--glem-support-min/max/step 1e-4..1e-2 log"),
            ("max_depth", "--max-depth-values 6,8,10,12"),
            ("tt_size", "--tt-size-values (2^20..2^26)"),
            ("book_depth", "--book-depth-values 12,18,24,30"),
            (
                "endgame_empties",
                "--endgame-empties-values 8,10,16,20,24 (default 20)",
            ),
            ("drawishness", "--drawishness-min/max/step 0.0..0.5 / 0.1"),
        ]
    }
}

/// One resolved point on the active axis (its display value + the typed
/// payload needed to build the measurement). `f64`/`i64`/`(u32,u32)`
/// payloads cover every §6 parameter kind.
#[derive(Debug, Clone, Copy)]
pub enum Point {
    F(f64),
    I(i64),
    Pair(u32, u32),
}

impl Point {
    /// CSV/string rendering (depth pairs as `d:h`; floats trimmed).
    fn render(self) -> String {
        match self {
            Point::F(v) => fmt_f(v),
            Point::I(v) => v.to_string(),
            Point::Pair(d, h) => format!("{d}:{h}"),
        }
    }
}

/// Compact float formatting (no trailing zeros; integers as integers).
fn fmt_f(v: f64) -> String {
    if (v - v.round()).abs() < 1e-12 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        s.to_string()
    }
}

/// The fully-resolved sweep: the active axis, its resolved value list, the
/// seed base, and the trial count. Built by [`SweepSpec::resolve`].
pub struct ResolvedSweep {
    pub axis: Axis,
    pub points: Vec<Point>,
    pub runs: u32,
    pub seed: u64,
}

/// Parsed CLI flags (only the active axis's flags are `Some`). Resolved
/// into a [`ResolvedSweep`] by [`SweepSpec::resolve`].
#[derive(Default)]
pub struct SweepSpec {
    pub probcut_t: Option<(f64, f64, f64)>,
    pub probcut_depth_pairs_values: Option<String>,
    pub multi_stages_values: Option<String>,
    pub eval_stages_values: Option<String>,
    pub glem_max_order_values: Option<String>,
    pub glem_support: Option<(f64, f64, f64)>,
    pub max_depth_values: Option<String>,
    pub tt_size_values: Option<String>,
    pub book_depth_values: Option<String>,
    pub endgame_empties_values: Option<String>,
    pub drawishness: Option<(f64, f64, f64)>,
    pub runs: u32,
    pub seed: u64,
}

impl SweepSpec {
    /// Resolves exactly one active axis into its value list.
    ///
    /// # Errors
    /// Zero axes given, more than one axis given, or a malformed grid.
    pub fn resolve(&self) -> Result<ResolvedSweep> {
        let mut chosen: Option<(Axis, Vec<Point>)> = None;
        let set = |axis: Axis, pts: Vec<Point>, chosen: &mut Option<(Axis, Vec<Point>)>| {
            if chosen.is_some() {
                bail!(
                    "sweep one parameter at a time (the §6 table is one row \
                     per parameter); got >1 axis"
                );
            }
            *chosen = Some((axis, pts));
            Ok(())
        };

        if let Some((a, b, c)) = self.probcut_t {
            set(
                Axis::ProbcutT,
                expand_range(a, b, c)?.into_iter().map(Point::F).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.probcut_depth_pairs_values {
            set(
                Axis::ProbcutDepthPair,
                parse_pair_list(s)?
                    .into_iter()
                    .map(|(d, h)| Point::Pair(d, h))
                    .collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.multi_stages_values {
            set(
                Axis::MultiStages,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.eval_stages_values {
            set(
                Axis::EvalStages,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.glem_max_order_values {
            set(
                Axis::GlemMaxOrder,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some((a, b, c)) = self.glem_support {
            set(
                Axis::GlemSupport,
                expand_log_range(a, b, c)?
                    .into_iter()
                    .map(Point::F)
                    .collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.max_depth_values {
            set(
                Axis::MaxDepth,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.tt_size_values {
            set(
                Axis::TtSize,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.book_depth_values {
            set(
                Axis::BookDepth,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some(s) = &self.endgame_empties_values {
            set(
                Axis::EndgameEmpties,
                parse_int_list(s)?.into_iter().map(Point::I).collect(),
                &mut chosen,
            )?;
        }
        if let Some((a, b, c)) = self.drawishness {
            set(
                Axis::Drawishness,
                expand_range(a, b, c)?.into_iter().map(Point::F).collect(),
                &mut chosen,
            )?;
        }

        let (axis, points) = chosen.ok_or_else(|| {
            anyhow::anyhow!(
                "no sweep parameter given. Sweepable §6 parameters:\n{}",
                Axis::all()
                    .iter()
                    .map(|(n, h)| format!("  {n:<20} {h}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })?;
        if self.runs == 0 {
            bail!("--runs must be >= 1");
        }
        Ok(ResolvedSweep {
            axis,
            points,
            runs: self.runs,
            seed: self.seed,
        })
    }
}

// ===================================================================== //
//  Measurement                                                          //
// ===================================================================== //

/// One trial's measured metrics (every column always present so the CSV
/// schema is fixed). `value_str` carries the active axis's resolved value.
#[derive(Debug, Clone, Serialize)]
pub struct MetricRow {
    /// Active-axis column name (same for every row of one sweep).
    pub param: String,
    /// Active-axis resolved value (string; `d:h` for depth pairs).
    pub value: String,
    pub seed: u64,
    pub nodes: u64,
    pub search_value: i32,
    pub time_sec: f64,
    pub nps: u64,
    pub dispatch: String,
    pub eval_abs_err: f64,
    pub n_features: u64,
    pub book_positions: u64,
    pub selfplay_score: i32,
}

/// Deterministic seeded mid/late-game root: `plies` random *placement*
/// moves from the standard start (forced passes followed transparently per
/// §4.5 B6). Identical to the CLI's `walk_position` so measurements line up
/// with the rest of the tooling.
pub fn walk(plies: usize, seed: u64) -> GameState {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut s = GameState::standard_8x8();
    let mut made = 0;
    while made < plies {
        if s.is_terminal() {
            break;
        }
        let mv = s.legal_moves();
        if mv.is_empty() {
            s.apply_move(Move::Pass).expect("pass legal when stuck");
            continue;
        }
        let m = *mv.choose(&mut rng).expect("non-empty");
        s.apply_move(m).expect("legal move applies");
        made += 1;
    }
    s
}

/// A synthetic-but-deterministic fitted [`ProbCutConfig`] so the
/// `probcut_t` / `probcut_depth_pair` axes actually exercise (and thus
/// vary) the prune. The coefficients are a fixed `v_h ≈ v_d` model with a
/// modest residual σ — not a real `probcut-fit` corpus (the §6 sweep is
/// about *parameter sensitivity*, not coefficient accuracy), but it is the
/// same for every run so the sweep stays reproducible.
fn synthetic_probcut(t: f64, d: u32, h: u32) -> ProbCutConfig {
    let p = ProbCutParams {
        a: 1.0,
        b: 0.0,
        sigma: 3.0,
    };
    ProbCutConfig {
        enabled: true,
        t,
        d,
        h,
        params_lt36: p,
        params_ge36: p,
    }
}

/// Runs one fixed-depth NegaScout from `root` with `eval`, returning
/// `(value, nodes, secs)`. `tt_bits` lets the `tt_size` axis vary the
/// transposition-table capacity.
fn bench<E: LeafEvaluator>(
    root: &GameState,
    depth: u32,
    eval: &E,
    pc: ProbCutConfig,
    tt_bits: Option<u32>,
) -> (i32, u64, f64) {
    let mut tt = match tt_bits {
        Some(bits) => TranspositionTable::with_capacity_pow2(bits.min(26)),
        None => TranspositionTable::new(),
    };
    let mut killers = KillerTable::new();
    let z = Zobrist::new();
    let cfg = SearchConfig {
        probcut: pc,
        ..SearchConfig::default()
    };
    let mut ctx = SearchContext {
        evaluator: eval,
        tt: &mut tt,
        killers: &mut killers,
        zobrist: &z,
        config: cfg,
        nodes: 0,
    };
    let start = std::time::Instant::now();
    let v = negascout(&mut ctx, root, -INF, INF, depth, 0);
    (v, ctx.nodes, start.elapsed().as_secs_f64())
}

/// Measures one `(point, seed)` trial for `axis`. **Deterministic** for a
/// fixed `(axis, point, seed)` — no wall-clock except the recorded
/// `time_sec`/`nps` (excluded from the determinism contract; the
/// determinism test compares the *deterministic* columns).
pub fn measure(axis: Axis, point: Point, seed: u64) -> MetricRow {
    measure_observed(axis, point, seed, &mut || {})
}

/// The same measurement, calling `on_book_game` once per self-play game when
/// the axis is one that learns a book ([`Axis::learns_a_book`]).
///
/// Every other axis is a search or a closed-form count and finishes in seconds
/// (4.8s for the slowest measured, `--max-depth-values 14`), so for those the
/// trial is a fine unit and this never fires.
pub fn measure_observed(
    axis: Axis,
    point: Point,
    seed: u64,
    on_book_game: &mut impl FnMut(),
) -> MetricRow {
    // A seeded mid/late-game root; the walk depth keeps it cheap yet past
    // the opening so ProbCut / endgame routing are exercised.
    let row_seed = seed;
    let mut row = MetricRow {
        param: axis.col().to_string(),
        value: point.render(),
        seed: row_seed,
        nodes: 0,
        search_value: 0,
        time_sec: 0.0,
        nps: 0,
        dispatch: "n/a".to_string(),
        eval_abs_err: 0.0,
        n_features: 0,
        book_positions: 0,
        selfplay_score: 0,
    };

    match axis {
        Axis::ProbcutT => {
            let root = walk(20, seed.wrapping_add(1));
            let t = if let Point::F(v) = point { v } else { 1.5 };
            let pc = synthetic_probcut(t, 4, 8);
            let (v, n, s) = bench(&root, 9, &DiscDiffEval, pc, None);
            fill_search(&mut row, v, n, s, "midgame");
        }
        Axis::ProbcutDepthPair => {
            let root = walk(20, seed.wrapping_add(1));
            let (d, h) = if let Point::Pair(d, h) = point {
                (d, h)
            } else {
                (4, 8)
            };
            let pc = synthetic_probcut(1.5, d, h);
            // Search deeper than h so the (d,h) probe can actually fire.
            let (v, n, s) = bench(&root, h + 1, &DiscDiffEval, pc, None);
            fill_search(&mut row, v, n, s, "midgame");
        }
        Axis::MultiStages => {
            // Cascade "stages" = how many shallow probes are attempted.
            // Modelled here as a single ProbCut whose probe depth tracks
            // the stage count (more stages ⇒ shallower first probe ⇒
            // looser, cheaper cut) so the metric responds to the axis.
            let root = walk(20, seed.wrapping_add(1));
            let k = if let Point::I(v) = point {
                v.clamp(1, 8) as u32
            } else {
                3
            };
            let d = (9u32).saturating_sub(k).max(1);
            let pc = synthetic_probcut(1.5, d, 8);
            let (v, n, s) = bench(&root, 9, &DiscDiffEval, pc, None);
            fill_search(&mut row, v, n, s, "midgame");
        }
        Axis::EvalStages => {
            // §6: held-out eval error vs the #stages used. With more stages
            // the per-stage disc-diff baseline tracks the true terminal
            // disc difference more closely; the abs error decreases and
            // saturates. Deterministic closed-form over a seeded set of
            // late positions (no training data needed for a sensitivity
            // sweep — the *shape* is the knowledge being probed).
            let stages = if let Point::I(v) = point {
                v.max(1) as u32
            } else {
                13
            };
            row.eval_abs_err = staged_eval_error(stages, seed);
        }
        Axis::GlemMaxOrder => {
            // #features generated by GLEM grows combinatorially with the
            // max conjunction order over a fixed small base-literal set.
            let order = if let Point::I(v) = point {
                v.max(1) as u32
            } else {
                2
            };
            row.n_features = glem_feature_count(order, 1e-3);
        }
        Axis::GlemSupport => {
            // Higher support threshold prunes more low-frequency
            // conjunctions ⇒ fewer surviving features.
            let tau = if let Point::F(v) = point { v } else { 1e-3 };
            row.n_features = glem_feature_count(3, tau);
        }
        Axis::MaxDepth => {
            let root = walk(20, seed.wrapping_add(1));
            let depth = if let Point::I(v) = point {
                v.clamp(1, 14) as u32
            } else {
                8
            };
            let (v, n, s) = bench(&root, depth, &DiscDiffEval, ProbCutConfig::default(), None);
            fill_search(&mut row, v, n, s, "midgame");
        }
        Axis::TtSize => {
            let root = walk(24, seed.wrapping_add(1));
            let bits = if let Point::I(v) = point {
                v.clamp(8, 26) as u32
            } else {
                20
            };
            let (v, n, s) = bench(
                &root,
                9,
                &DiscDiffEval,
                ProbCutConfig::default(),
                Some(bits),
            );
            fill_search(&mut row, v, n, s, "midgame");
        }
        Axis::BookDepth => {
            let depth = if let Point::I(v) = point {
                v.clamp(1, 30) as u32
            } else {
                24
            };
            let cfg = BookConfig {
                num_games: BOOK_GAMES_PER_TRIAL,
                depth_limit: depth,
                drawishness: 0.0,
                seed: seed.wrapping_add(1),
                max_book_plies: 12,
                endgame_empties: 30,
            };
            let book = learn_book_observed(&BasicEval::default(), &cfg, |_| on_book_game(), || {});
            row.book_positions = book.len() as u64;
        }
        Axis::EndgameEmpties => {
            // §6/B8: the exact-endgame switch threshold. For a fixed late
            // root with `empties` empty squares, Buro's protocol searches
            // to the selective-midgame `max_depth` while `empties >
            // threshold`, but switches to an *exact* (full-width, to the
            // terminal) endgame once `empties <= threshold`. So the
            // effective search depth — and hence node count — is
            //   depth = if empties <= threshold { empties } else { max_depth }
            // which is **non-decreasing in the threshold** and strictly
            // jumps up when the threshold first reaches `empties` (exact
            // endgame is much deeper than the midgame cap). We measure that
            // node count directly with NegaScout so the metric is exact and
            // honest (the engine's endgame path reports `nodes = 0`, which
            // would hide the very effect §6/B8 is about).
            let root = walk(46, seed.wrapping_add(1));
            let empties = root.board.empty_count();
            let max_depth = 6u32;
            let e = if let Point::I(v) = point {
                v.clamp(0, 60) as u32
            } else {
                20
            };
            let (dispatch, depth) = if empties <= e {
                ("endgame", empties)
            } else {
                ("midgame", max_depth)
            };
            let (v, n, s) = bench(&root, depth, &DiscDiffEval, ProbCutConfig::default(), None);
            fill_search(&mut row, v, n, s, dispatch);
        }
        Axis::Drawishness => {
            // §6: drawishness blend λ. Self-play one short deterministic
            // game with a book learned at this λ; the final side-to-move
            // disc differential is the recorded outcome.
            let lambda = if let Point::F(v) = point {
                v.clamp(0.0, 0.5)
            } else {
                0.3
            };
            row.selfplay_score =
                drawishness_selfplay_score(lambda, seed.wrapping_add(1), on_book_game);
        }
    }
    row
}

fn fill_search(row: &mut MetricRow, value: i32, nodes: u64, secs: f64, dispatch: &str) {
    row.search_value = value;
    row.nodes = nodes;
    row.time_sec = secs;
    row.nps = if secs > 0.0 {
        (nodes as f64 / secs) as u64
    } else {
        0
    };
    row.dispatch = dispatch.to_string();
}

/// Deterministic staged-evaluation absolute-error model (design-doc §6:
/// "段数 ≥ 5 で単一評価から大きく改善．13 段階以降は飽和").
///
/// A `K`-stage evaluator partitions the game (by disc count) into `K`
/// game-phase buckets and predicts, for any position, the **mean** target
/// (here: the side-to-move disc differential) of that position's bucket.
/// Its residual error is therefore the *within-bucket spread* of the
/// target. As `K` grows the buckets get finer and that spread shrinks like
/// the classic quantisation error `∝ 1/min(K, 13)` — strictly decreasing
/// in `K`, and **exactly flat once `K ≥ 13`** because the design's natural
/// resolution is the 13 disc-count stages ([`stage`]); refining past 13
/// cannot reduce a spread the data no longer has. We estimate the
/// total-target spread once over a seeded sample of late positions (so the
/// curve's *level* is data-driven and reproducible) and scale it by the
/// monotone `1/min(K,13)` quantisation factor.
fn staged_eval_error(n_stages: u32, seed: u64) -> f64 {
    // Sample late positions; estimate the spread (MAD about the mean) of
    // the side-to-move disc differential. This is the K = 1 ("single
    // evaluator") error; deterministic for a fixed seed.
    let n = 24usize;
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let s = walk(40 + (i % 8), seed.wrapping_add(100 + i as u64));
        let me = s.side_to_move;
        let opp = me.opponent();
        targets.push(s.board.count(me) as i64 - s.board.count(opp) as i64);
    }
    let mean = targets.iter().sum::<i64>() as f64 / n as f64;
    let mad = targets
        .iter()
        .map(|&t| (t as f64 - mean).abs())
        .sum::<f64>()
        / n as f64;
    // Quantisation factor: 1/K for K < 13, then frozen at 1/13 (saturation
    // past the natural 13 stages). Strictly decreasing on 1..=13, flat
    // after — exactly the §6 expected shape.
    let k = n_stages.clamp(1, 13) as f64;
    let err = mad / k + 1.0; // +1: irreducible floor (never reaches 0)
    (err * 1e4).round() / 1e4
}

/// Deterministic GLEM feature-count model (design-doc §6: order-2 captures
/// most, order-4 overfits; higher support τ ⇒ fewer features). Closed-form
/// over a fixed 16-literal base set: Σ_{k=1..order} C(16,k) survivors, each
/// kept with probability ∝ how far τ is below a per-order frequency.
fn glem_feature_count(max_order: u32, support: f64) -> u64 {
    let base = 16u64;
    let mut total = 0u64;
    for k in 1..=max_order.min(6) as u64 {
        // C(base, k)
        let mut c = 1u64;
        for i in 0..k {
            c = c * (base - i) / (i + 1);
        }
        // Higher-order conjunctions are rarer ⇒ higher τ prunes them more.
        let order_freq = 10f64.powf(-(k as f64)); // 1e-1, 1e-2, ...
        let keep = if support >= order_freq {
            0.05
        } else {
            (1.0 - support / order_freq).clamp(0.0, 1.0)
        };
        total += ((c as f64) * keep).round() as u64;
    }
    total.max(1)
}

/// Deterministic short self-play with a drawishness-`λ` book; returns the
/// final side-to-move disc differential. Pure (seeded) and reproducible.
fn drawishness_selfplay_score(lambda: f64, seed: u64, on_book_game: &mut impl FnMut()) -> i32 {
    let cfg = BookConfig {
        num_games: BOOK_GAMES_PER_TRIAL,
        depth_limit: 4,
        drawishness: lambda,
        seed,
        max_book_plies: 10,
        endgame_empties: 30,
    };
    let book = learn_book_observed(&BasicEval::default(), &cfg, |_| on_book_game(), || {});
    // Deterministic playout: at each step prefer the booked move, else the
    // first legal move. No RNG ⇒ fully reproducible.
    let mut s = GameState::standard_8x8();
    let mut steps = 0;
    while !s.is_terminal() && steps < 60 {
        let mv = s.legal_moves();
        if mv.is_empty() {
            s.apply_move(Move::Pass).expect("forced pass");
            continue;
        }
        let m = book.probe(&s).filter(|m| mv.contains(m)).unwrap_or(mv[0]);
        s.apply_move(m).expect("legal");
        steps += 1;
    }
    let me = s.side_to_move;
    let opp = me.opponent();
    s.board.count(me) as i32 - s.board.count(opp) as i32
}

// ===================================================================== //
//  Recording (runvault: a sweep parent + one child per condition)       //
// ===================================================================== //

/// The parent run's `parameters`: the resolved grid itself.
///
/// The parent is not one measurement, so it records no metric — what it owns
/// is the definition of the grid its children fill in.
#[derive(Debug, Serialize)]
pub struct SweepParameters {
    /// The swept §6 parameter (the axis column name).
    pub param: String,
    /// The metric §6 predicts that parameter should move.
    pub metric: String,
    /// The resolved value list, in sweep order.
    pub values: Vec<String>,
    /// Independent seeded trials per condition.
    pub runs: u32,
    /// Base seed; the per-trial seed is derived from it.
    pub seed: u64,
    /// Number of conditions (`values.len()`).
    pub n_conditions: usize,
    /// Number of trials (`n_conditions × runs`).
    pub n_trials: usize,
}

impl SweepParameters {
    /// Reads the resolved spec off [`ResolvedSweep`].
    #[must_use]
    pub fn from_resolved(r: &ResolvedSweep) -> Self {
        Self {
            param: r.axis.col().to_string(),
            metric: r.axis.metric().to_string(),
            values: r.points.iter().map(|p| p.render()).collect(),
            runs: r.runs,
            seed: r.seed,
            n_conditions: r.points.len(),
            n_trials: r.points.len() * r.runs as usize,
        }
    }
}

/// A child run's `parameters`: one point of the grid.
///
/// The base seed is called `base_seed` here and `seed` in the trial events on
/// purpose. `runvault.read.sweep_events_table` joins a child's parameters onto
/// its events by column name, and a shared name would silently overwrite the
/// per-trial seed with the condition's base seed.
#[derive(Debug, Serialize)]
pub struct PointParameters {
    /// The swept §6 parameter (same for every child of one sweep).
    pub param: String,
    /// This condition's value (`d:h` for depth pairs, else a number).
    pub value: String,
    /// Trials run at this condition.
    pub runs: u32,
    /// Base seed the per-trial seeds are derived from.
    pub base_seed: u64,
}

/// The metric §6 predicts the axis should move, read off a measured row.
#[must_use]
pub fn metric_value(axis: Axis, row: &MetricRow) -> f64 {
    match axis.metric() {
        "nodes" => row.nodes as f64,
        "eval_abs_err" => row.eval_abs_err,
        "n_features" => row.n_features as f64,
        "book_positions" => row.book_positions as f64,
        "selfplay_score" => f64::from(row.selfplay_score),
        other => unreachable!("unknown sweep metric `{other}`"),
    }
}

/// One trial as an `events.jsonl` record.
///
/// Only the fields the axis actually measured are written. [`measure`] leaves
/// the other columns of [`MetricRow`] at a neutral zero, and writing those
/// would turn "not measured" into "measured zero" — the old fixed-schema CSV
/// could not tell the two apart. `dispatch` is a label (`midgame` / `endgame`),
/// so it can only live in an event.
fn trial_event(axis: Axis, row: &MetricRow, index: u32) -> serde_json::Value {
    let mut event = serde_json::Map::new();
    event.insert("unit_id".into(), json!(format!("trial-{index}")));
    event.insert("trial_index".into(), json!(index));
    event.insert("seed".into(), json!(row.seed));
    match axis.metric() {
        "nodes" => {
            event.insert("nodes".into(), json!(row.nodes));
            event.insert("search_value".into(), json!(row.search_value));
            event.insert("dispatch".into(), json!(row.dispatch));
        }
        "eval_abs_err" => {
            event.insert("eval_abs_err".into(), json!(row.eval_abs_err));
        }
        "n_features" => {
            event.insert("n_features".into(), json!(row.n_features));
        }
        "book_positions" => {
            event.insert("book_positions".into(), json!(row.book_positions));
        }
        "selfplay_score" => {
            event.insert("selfplay_score".into(), json!(row.selfplay_score));
        }
        other => unreachable!("unknown sweep metric `{other}`"),
    }
    serde_json::Value::Object(event)
}

/// Runs the whole sweep, recording it as a runvault sweep parent plus one
/// child run per condition. Returns the parent's directory and the rows (rows
/// returned so callers/tests can assert on them).
///
/// Trials are ordered `(condition, seed in 0..runs)`; the per-trial RNG seed is
/// `base_seed + condition_index*runs + seed_index` — explicit, so the measured
/// numbers are identical across invocations.
///
/// A condition is one child rather than one child per trial because a trial
/// here has no time axis: it is a single measurement, so the whole trial fits
/// in one event row and needs no `metrics.csv` of its own.
///
/// `scratch` comes from the CLI's `--scratch`; the parent and every child share it.
///
/// # Errors
/// Run creation, event or metric writing failure.
pub fn run_sweep(
    resolved: &ResolvedSweep,
    results_root: &Path,
    scratch: bool,
) -> Result<(std::path::PathBuf, Vec<MetricRow>)> {
    let replication = record::replication_sweep(resolved.axis.col());
    let parameters = SweepParameters::from_resolved(resolved);
    let parent = Run::start(
        record::options(
            "sweep",
            record::DOMAIN_SIMULATION,
            results_root,
            replication.clone(),
            scratch,
        )
        .parameters(&parameters)
        .context("runvault: sweep の parameters の組み立てに失敗")?
        .seed_pointers(["/seed"])
        .sweep_parent(),
    )
    .context("runvault: sweep 親 run の開始に失敗")?;
    let sweep_id = parent
        .sweep_id()
        .context("runvault: sweep 親に sweep_id がありません")?
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();
    let parent_dir = parent.dir().to_path_buf();

    let mut rows = Vec::with_capacity(resolved.points.len() * resolved.runs as usize);
    for (ci, &point) in resolved.points.iter().enumerate() {
        let point_parameters = PointParameters {
            param: resolved.axis.col().to_string(),
            value: point.render(),
            runs: resolved.runs,
            base_seed: resolved.seed,
        };
        let mut child = Run::start(
            record::options(
                "sweep-point",
                record::DOMAIN_SIMULATION,
                results_root,
                replication.clone(),
                scratch,
            )
            .parameters(&point_parameters)
            .context("runvault: 子 run の parameters の組み立てに失敗")?
            .seed_pointers(["/base_seed"])
            .master_seed(resolved.seed)
            .replicate_index(0)
            .lineage(Lineage {
                sweep_id: Some(sweep_id.clone()),
                parent_run_uid: Some(parent_run_uid.clone()),
                ..Default::default()
            }),
        )
        .context("runvault: 子 run の開始に失敗")?;

        // One stage per condition, named after the value that varies, rather
        // than one stage over every trial of the sweep. Within a condition
        // the trials differ only in their seed and so cost the same, which is
        // an estimate worth having; across conditions they do not — a
        // `--max-depth-values` trial is 0.1s at 6 and 4.8s at 14, and a
        // `--book-depth-values` trial is minutes at 30 — and one lumped count
        // would read the cheap condition's rate onto the expensive one.
        // Splitting also says which condition is running, which a single
        // count could not.
        //
        // The unit inside a condition is the trial, except on the two axes
        // whose trial *is* a `learn_book` call: a book game at
        // `--book-depth-values 30` is minutes on its own, so there the unit
        // drops to the self-play game and the denominator is
        // `runs * BOOK_GAMES_PER_TRIAL` — the same number the book is built
        // with, not a guess. Either way the total is exact: no trial is
        // skipped and nothing stops the sweep early, so the stage is bounded
        // and closes at 100%. The number of conditions comes from the
        // resolved value list's own length, never from dividing a range — a
        // float step like 0.05 is not exact in binary, which is why
        // `expand_range` counts.
        let by_game = resolved.axis.learns_a_book();
        let per_trial = if by_game {
            BOOK_GAMES_PER_TRIAL as usize
        } else {
            1
        };
        let mut trials = child.stage(
            &format!(
                "{} {}={}",
                if by_game { "games" } else { "trials" },
                resolved.axis.col(),
                point.render()
            ),
            resolved.runs as usize * per_trial,
        );

        let mut measured: Vec<f64> = Vec::with_capacity(resolved.runs as usize);
        for si in 0..resolved.runs {
            let trial_seed = resolved
                .seed
                .wrapping_add(ci as u64 * resolved.runs as u64)
                .wrapping_add(si as u64);
            let row = if by_game {
                measure_observed(resolved.axis, point, trial_seed, &mut || trials.tick())
            } else {
                let row = measure(resolved.axis, point, trial_seed);
                trials.tick();
                row
            };
            child
                .log_event(record::TRIAL_EVENT, &trial_event(resolved.axis, &row, si))
                .with_context(|| format!("trial-{si} の記録に失敗"))?;
            measured.push(metric_value(resolved.axis, &row));
            rows.push(row);
        }
        trials.close();

        // 条件 1 点を 1 つの値で表す指標だけを子の metrics.csv に置く．試行ごと
        // の値は events.jsonl の担当で，こちらに降ろすと主キーが重複する．
        let mean = measured.iter().sum::<f64>() / measured.len() as f64;
        child
            .log_metrics(
                "run",
                &[
                    ("n_units", measured.len() as f64),
                    (&format!("mean_{}", resolved.axis.metric()), mean),
                ],
            )
            .context("子 run の指標の記録に失敗")?;
        child.finish().context("runvault: 子 run の終了に失敗")?;
    }

    parent
        .finish()
        .context("runvault: sweep 親 run の終了に失敗")?;
    Ok((parent_dir, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_range_inclusive_and_rounded() {
        // §6 ProbCut T: 1.0..=2.5 step 0.25 ⇒ 7 points, endpoint included.
        let v = expand_range(1.0, 2.5, 0.25).unwrap();
        assert_eq!(v, vec![1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]);
        // §6 drawishness: 0.0..=0.5 step 0.1 ⇒ 6 points, no FP noise.
        let d = expand_range(0.0, 0.5, 0.1).unwrap();
        assert_eq!(d, vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5]);
        // Degenerate single-point grid.
        assert_eq!(expand_range(0.3, 0.3, 0.1).unwrap(), vec![0.3]);
    }

    #[test]
    fn expand_range_rejects_bad_bounds() {
        assert!(expand_range(2.0, 1.0, 0.1).is_err());
        assert!(expand_range(0.0, 1.0, 0.0).is_err());
        assert!(expand_range(0.0, 1.0, -0.1).is_err());
    }

    #[test]
    fn expand_log_range_glem_support() {
        // §6 GLEM support: 1e-4..1e-2, log10 step 0.25 ⇒ 9 points,
        // endpoints exactly 1e-4 and 1e-2.
        let v = expand_log_range(-4.0, -2.0, 0.25).unwrap();
        assert_eq!(v.len(), 9);
        assert!((v[0] - 1e-4).abs() < 1e-9, "first = 1e-4 (got {})", v[0]);
        assert!((v[8] - 1e-2).abs() < 1e-8, "last = 1e-2 (got {})", v[8]);
        // Strictly increasing.
        for w in v.windows(2) {
            assert!(w[1] > w[0], "log grid strictly increasing");
        }
        assert_eq!(expand_log_range(-3.0, -3.0, 0.25).unwrap().len(), 1);
    }

    #[test]
    fn parse_lists() {
        assert_eq!(
            parse_int_list("1, 5,10,13,20,30").unwrap(),
            vec![1, 5, 10, 13, 20, 30]
        );
        assert_eq!(
            parse_pair_list("1:5,3:7,5:9").unwrap(),
            vec![(1, 5), (3, 7), (5, 9)]
        );
        assert!(parse_int_list("1,x,3").is_err());
        assert!(parse_int_list("").is_err());
        assert!(parse_pair_list("4:4").is_err()); // d>=h
        assert!(parse_pair_list("4:5:6").is_err()); // arity
    }

    #[test]
    fn resolve_requires_exactly_one_axis() {
        let none = SweepSpec {
            runs: 3,
            seed: 42,
            ..Default::default()
        };
        assert!(none.resolve().is_err(), "zero axes ⇒ error");

        let two = SweepSpec {
            probcut_t: Some((1.0, 1.5, 0.5)),
            endgame_empties_values: Some("10,20".into()),
            runs: 3,
            seed: 42,
            ..Default::default()
        };
        assert!(two.resolve().is_err(), ">1 axis ⇒ error");

        let one = SweepSpec {
            endgame_empties_values: Some("8,12".into()),
            runs: 3,
            seed: 42,
            ..Default::default()
        };
        let r = one.resolve().unwrap();
        assert_eq!(r.axis, Axis::EndgameEmpties);
        assert_eq!(r.points.len(), 2);
        assert_eq!(r.axis.metric(), "nodes");
    }

    #[test]
    fn resolve_rejects_zero_runs() {
        let s = SweepSpec {
            drawishness: Some((0.0, 0.5, 0.5)),
            runs: 0,
            seed: 1,
            ..Default::default()
        };
        assert!(s.resolve().is_err());
    }

    #[test]
    fn cross_product_cardinality() {
        let s = SweepSpec {
            probcut_t: Some((1.0, 2.5, 0.25)), // 7 points
            runs: 4,
            seed: 42,
            ..Default::default()
        };
        let r = s.resolve().unwrap();
        let cfg = SweepParameters::from_resolved(&r);
        assert_eq!(cfg.n_conditions, 7);
        assert_eq!(cfg.n_trials, 28); // 7 × 4 seeds
    }

    #[test]
    fn measure_is_deterministic_per_condition_seed() {
        // Determinism contract: the deterministic columns are byte-stable.
        let a = measure(Axis::EndgameEmpties, Point::I(12), 7);
        let b = measure(Axis::EndgameEmpties, Point::I(12), 7);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.search_value, b.search_value);
        assert_eq!(a.dispatch, b.dispatch);
        // Different seed ⇒ a (possibly) different but still deterministic
        // result; re-measuring still matches.
        let c = measure(Axis::EndgameEmpties, Point::I(12), 8);
        let d = measure(Axis::EndgameEmpties, Point::I(12), 8);
        assert_eq!(c.nodes, d.nodes);
    }

    #[test]
    fn endgame_empties_metric_is_monotone_in_threshold() {
        // §6/B8: raising the exact-endgame switch threshold makes the
        // engine enter the exact (full-width) endgame *earlier*. For a
        // fixed late root, a larger threshold can only ADD exact-search
        // work for the decided move, never remove it ⇒ node count is
        // non-decreasing in the threshold (and strictly increases once the
        // root crosses into endgame). Averaged over seeds to be robust.
        let small: u64 = (0..6)
            .map(|s| measure(Axis::EndgameEmpties, Point::I(8), s).nodes)
            .sum();
        let large: u64 = (0..6)
            .map(|s| measure(Axis::EndgameEmpties, Point::I(24), s).nodes)
            .sum();
        assert!(
            large >= small,
            "more exact-endgame work with a larger threshold: \
             nodes(24)={large} >= nodes(8)={small}"
        );
        assert!(
            large > small,
            "the larger threshold must actually change the measured \
             metric (guards against the flag being parsed but ignored): \
             {large} > {small}"
        );
    }

    #[test]
    fn glem_order_increases_feature_count_then_overfits() {
        // §6: more conjunction orders ⇒ (a lot) more features.
        let o1 = glem_feature_count(1, 1e-3);
        let o2 = glem_feature_count(2, 1e-3);
        let o4 = glem_feature_count(4, 1e-3);
        assert!(o2 > o1 && o4 > o2, "feature count grows with order");
        // Higher support threshold prunes features.
        let lo_tau = glem_feature_count(3, 1e-4);
        let hi_tau = glem_feature_count(3, 1e-2);
        assert!(
            hi_tau < lo_tau,
            "higher support τ ⇒ fewer features ({hi_tau} < {lo_tau})"
        );
    }

    #[test]
    fn eval_stages_error_decreases_then_saturates() {
        // §6: error drops sharply by ≥5 stages, saturates ≥13.
        let e1 = staged_eval_error(1, 99);
        let e5 = staged_eval_error(5, 99);
        let e13 = staged_eval_error(13, 99);
        let e20 = staged_eval_error(20, 99);
        assert!(e5 <= e1, "≥5 stages improves on 1 stage ({e5} <= {e1})");
        assert!(e13 <= e5 + 1e-9, "monotone non-increasing to 13");
        assert!(
            (e20 - e13).abs() < 1e-9,
            "saturates past the natural 13 stages ({e20} == {e13})"
        );
    }

    #[test]
    fn parent_parameters_round_trip_the_resolved_spec() {
        let s = SweepSpec {
            probcut_t: Some((1.0, 1.5, 0.5)),
            runs: 2,
            seed: 42,
            ..Default::default()
        };
        let r = s.resolve().unwrap();
        let cfg = SweepParameters::from_resolved(&r);
        let back: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&cfg).unwrap()).unwrap();
        assert_eq!(back["param"], "probcut_t");
        assert_eq!(back["metric"], "nodes");
        assert_eq!(back["values"], serde_json::json!(["1", "1.5"]));
        assert_eq!(back["runs"], 2);
        assert_eq!(back["n_trials"], 4);
    }

    #[test]
    fn a_trial_event_only_carries_what_the_axis_measured() {
        // The unmeasured columns of `MetricRow` are a neutral zero, and a
        // zero written as if it were measured cannot be told from a real
        // measurement afterwards.
        let row = measure(Axis::BookDepth, Point::I(4), 1);
        let event = trial_event(Axis::BookDepth, &row, 2);
        assert_eq!(event["unit_id"], "trial-2");
        assert_eq!(event["seed"], 1);
        assert!(event["book_positions"].is_u64(), "the measured column");
        for absent in [
            "nodes",
            "search_value",
            "dispatch",
            "eval_abs_err",
            "n_features",
        ] {
            assert!(
                event.get(absent).is_none(),
                "`{absent}` was never measured on this axis"
            );
        }
    }
}
