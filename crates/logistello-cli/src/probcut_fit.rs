//! ProbCut coefficient estimation (`probcut-fit`; design doc §4.3.4 /
//! §4.5 B7; Phase 5).
//!
//! Buro fits `v_h = a·v_d + b + ε`, `ε ~ N(0, σ²)` from a corpus of
//! positions evaluated by the program's own brute-force search, **per
//! (disc-count phase, h, d)** (design doc §4.5 B7). This module reproduces
//! the single-pair `(d, h) = (4, 8)` fit:
//!
//! 1. Generate sample positions — seeded `RandomPlayer` self-play (no
//!    external data; reuses the Phase-4b `extract` corpus generator) or
//!    real WTHOR `.wtb` games.
//! 2. For every non-terminal sampled position run `v_d = NegaScout(d)` and
//!    `v_h = NegaScout(h)` with the chosen evaluator (`PatternEval` if a
//!    weight file is given, else `BasicEval`) under the **same TT
//!    discipline as production** (ProbCut OFF, TT + killers on — exactly
//!    the engine's selective-midgame search).
//! 3. Stratify by disc phase (`< 36` vs `≥ 36` discs, the production split
//!    of design doc §4.5 B7).
//! 4. Per phase fit ordinary least squares `v_h = a·v_d + b`, residual
//!    `σ = sqrt(SSE / (n − 2))`, and the coefficient of determination R².
//! 5. Emit a [`ProbCutConfig`]-compatible JSON file and print per-phase
//!    `a, b, σ, R², n`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use logistello_core::Zobrist;
use logistello_eval::{BasicEval, EvalWeights, LeafEvaluator, PatternEval};
use logistello_search::alphabeta::{INF, SearchConfig, SearchContext, negascout};
use logistello_search::killer::KillerTable;
use logistello_search::tt::TranspositionTable;
use logistello_search::{PROBCUT_PHASE_SPLIT_DISCS, ProbCutConfig, ProbCutParams};
use othello_core::{GameState, Move};
use othello_engine::{EngineConfig as GameEngineConfig, GameEngine};
use othello_io::WthorReader;
use othello_player::RandomPlayer;

/// One fitted disc-count phase: the OLS line, residual σ, R² and sample n.
#[derive(Debug, Clone, Copy)]
pub struct PhaseFit {
    /// Slope `a`.
    pub a: f64,
    /// Intercept `b`.
    pub b: f64,
    /// Residual standard deviation `σ = sqrt(SSE / (n − 2))`.
    pub sigma: f64,
    /// Coefficient of determination `R² = 1 − SSE/SST`.
    pub r2: f64,
    /// Number of `(v_d, v_h)` samples in this phase.
    pub n: usize,
}

impl PhaseFit {
    /// The [`ProbCutParams`] this fit implies.
    #[must_use]
    pub fn params(&self) -> ProbCutParams {
        ProbCutParams::new(self.a, self.b, self.sigma)
    }
}

/// Result of a single-pair `probcut-fit` run.
#[derive(Debug, Clone, Copy)]
pub struct FitResult {
    /// Shallow check depth `d`.
    pub d: u32,
    /// Deep height `h`.
    pub h: u32,
    /// Fit for the `< 36` discs phase.
    pub lt36: PhaseFit,
    /// Fit for the `≥ 36` discs phase.
    pub ge36: PhaseFit,
}

impl FitResult {
    /// Builds a [`ProbCutConfig`] from this fit (`enabled = true`, the
    /// canonical `T = 1.5` unless overridden by the caller after load).
    #[must_use]
    pub fn to_config(&self, t: f64) -> ProbCutConfig {
        ProbCutConfig {
            enabled: true,
            t,
            d: self.d,
            h: self.h,
            params_lt36: self.lt36.params(),
            params_ge36: self.ge36.params(),
        }
    }
}

/// Ordinary least squares of `y = a·x + b` plus residual σ and R².
///
/// `σ = sqrt(SSE / (n − 2))` (unbiased residual sd, 2 fitted params).
/// `R² = 1 − SSE/SST`. Degenerate inputs (`n < 3` or zero x-variance,
/// e.g. all probes saturated to the same value) yield a *non-usable*
/// fit (`a = 0`) so the search falls through rather than cut on noise.
fn ols(x: &[f64], y: &[f64]) -> PhaseFit {
    let n = x.len();
    if n < 3 {
        return PhaseFit {
            a: 0.0,
            b: 0.0,
            sigma: 0.0,
            r2: 0.0,
            n,
        };
    }
    let nf = n as f64;
    let mean_x = x.iter().sum::<f64>() / nf;
    let mean_y = y.iter().sum::<f64>() / nf;
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 1e-12 {
        // No variance in the shallow probe → cannot fit a line.
        return PhaseFit {
            a: 0.0,
            b: mean_y,
            sigma: (syy / (nf - 2.0)).sqrt(),
            r2: 0.0,
            n,
        };
    }
    let a = sxy / sxx;
    let b = mean_y - a * mean_x;
    // SSE = Σ(y - (a x + b))² = Syy - a·Sxy  (standard OLS identity).
    let sse = (syy - a * sxy).max(0.0);
    let sigma = (sse / (nf - 2.0)).sqrt();
    let r2 = if syy <= 1e-12 {
        // Constant y: perfectly explained iff SSE≈0.
        if sse <= 1e-9 { 1.0 } else { 0.0 }
    } else {
        1.0 - sse / syy
    };
    PhaseFit { a, b, sigma, r2, n }
}

/// Corpus source for `probcut-fit`.
#[derive(Debug, Clone)]
pub enum Source {
    /// Seeded `RandomPlayer` self-play (no external data).
    Selfplay {
        /// Number of self-play games to draw positions from.
        games: usize,
        /// Deterministic RNG seed.
        seed: u64,
    },
    /// Real expert games from `.wtb` files in a directory.
    Wthor {
        /// Directory containing `.wtb` files.
        dir: PathBuf,
        /// Max games to read.
        max_games: usize,
    },
}

/// Collects up to `max_samples` non-terminal positions from the corpus,
/// calling `on_game` once per corpus game consumed.
///
/// Self-play reuses the exact deterministic seeding of the Phase-4b
/// extractor; WTHOR replays each `.wtb` game (passes auto-inserted by the
/// reader, design doc §4.5 B6). Positions are taken in game order.
///
/// Both arms stop as soon as `max_samples` positions are in hand, and the
/// WThor arm also stops when the `.wtb` files run out, so neither `games` nor
/// `max_games` is a total the collection is bound to reach: this is counted in
/// an *unbounded* stage.
fn collect_positions_observed(
    src: &Source,
    max_samples: usize,
    mut on_game: impl FnMut(usize),
) -> Result<Vec<GameState>> {
    let mut out = Vec::new();
    match src {
        Source::Selfplay { games, seed } => {
            for g in 0..*games {
                if out.len() >= max_samples {
                    break;
                }
                let bseed = seed
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(g as u64);
                let wseed = bseed ^ 0xD1B5_4A32_D192_ED03;
                let mut engine = GameEngine::new(GameEngineConfig::standard())
                    .map_err(|e| anyhow::anyhow!("engine build: {e}"))?;
                let mut black = RandomPlayer::with_seed(othello_core::Color::Black, bseed);
                let mut white = RandomPlayer::with_seed(othello_core::Color::White, wseed);
                engine
                    .run(&mut black, &mut white)
                    .map_err(|e| anyhow::anyhow!("self-play game {g}: {e}"))?;
                for s in engine.history().snapshots() {
                    if s.is_terminal() {
                        continue;
                    }
                    out.push(s.clone());
                    if out.len() >= max_samples {
                        break;
                    }
                }
                on_game(g);
            }
        }
        Source::Wthor { dir, max_games } => {
            let mut wtb: Vec<_> = std::fs::read_dir(dir)
                .with_context(|| format!("reading WTHOR dir {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wtb")))
                .collect();
            wtb.sort();
            if wtb.is_empty() {
                bail!("no .wtb files found in {}", dir.display());
            }
            let mut games_done = 0usize;
            'outer: for path in wtb {
                let f = std::fs::File::open(&path)
                    .with_context(|| format!("open {}", path.display()))?;
                let mut reader = WthorReader::new(f)
                    .map_err(|e| anyhow::anyhow!("WTHOR header {}: {e}", path.display()))?;
                let records = reader
                    .read_all()
                    .map_err(|e| anyhow::anyhow!("WTHOR read {}: {e}", path.display()))?;
                for rec in records {
                    let moves: Vec<Move> = rec.moves.iter().map(|m| m.r#move).collect();
                    let mut st = GameState::standard_8x8();
                    for &mv in &moves {
                        if st.is_terminal() {
                            break;
                        }
                        if !st.is_terminal() {
                            out.push(st.clone());
                        }
                        if matches!(mv, Move::Place(_)) && st.must_pass() {
                            st.apply_move(Move::Pass).ok();
                        }
                        if st.apply_move(mv).is_err() {
                            break;
                        }
                        if out.len() >= max_samples {
                            break 'outer;
                        }
                    }
                    games_done += 1;
                    on_game(games_done);
                    if games_done >= *max_games {
                        break 'outer;
                    }
                }
            }
        }
    }
    out.truncate(max_samples);
    Ok(out)
}

/// Full-window NegaScout value at `depth` with production TT discipline
/// (TT + killers on, **ProbCut OFF** — exactly the engine's
/// selective-midgame search; a *fresh* TT per position so the fit is not
/// contaminated by cross-position TT carry-over).
fn search_value<E: LeafEvaluator>(s: &GameState, depth: u32, eval: &E) -> i32 {
    let mut tt = TranspositionTable::new();
    let mut killers = KillerTable::new();
    let z = Zobrist::new();
    let mut ctx = SearchContext {
        evaluator: eval,
        tt: &mut tt,
        killers: &mut killers,
        zobrist: &z,
        // ProbCut OFF: the fit corpus must be the *true* depth-d / depth-h
        // values, never ProbCut-approximated ones.
        config: SearchConfig::default(),
        nodes: 0,
    };
    negascout(&mut ctx, s, -INF, INF, depth, 0)
}

/// Runs the single-pair `(d, h)` fit over the corpus with `eval`, calling
/// `on_position` once per corpus position.
///
/// Every position is searched twice — once at `d` and once at `h` — so the
/// positions cost the same to within their own branching and the corpus is
/// counted, not weighted. `positions.len()` is exact: the loop skips a
/// terminal position but still counts it as tried.
fn fit_with_observed<E: LeafEvaluator>(
    positions: &[GameState],
    d: u32,
    h: u32,
    eval: &E,
    mut on_position: impl FnMut(usize, usize),
) -> FitResult {
    let mut lt_x = Vec::new();
    let mut lt_y = Vec::new();
    let mut ge_x = Vec::new();
    let mut ge_y = Vec::new();
    for (i, s) in positions.iter().enumerate() {
        // A position skipped for being terminal was still tried, so it ticks
        // before the `continue` rather than after it. The corpus size travels
        // with the index: it is not known until the collection has finished,
        // and it is the fit stage's denominator.
        on_position(i, positions.len());
        if s.is_terminal() {
            continue;
        }
        let discs = 64 - s.board.empty_count();
        let vd = f64::from(search_value(s, d, eval));
        let vh = f64::from(search_value(s, h, eval));
        if discs < PROBCUT_PHASE_SPLIT_DISCS {
            lt_x.push(vd);
            lt_y.push(vh);
        } else {
            ge_x.push(vd);
            ge_y.push(vh);
        }
    }
    FitResult {
        d,
        h,
        lt36: ols(&lt_x, &lt_y),
        ge36: ols(&ge_x, &ge_y),
    }
}

/// Public entry point: collect the corpus, run the `(d, h)` fit with the
/// chosen evaluator, write the JSON config, and return the [`FitResult`]
/// for printing.
///
/// # Errors
///
/// Propagates corpus / I/O errors.
#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    d: u32,
    h: u32,
    src: &Source,
    samples: usize,
    eval_weights: Option<&Path>,
    t: f64,
    output: &Path,
) -> Result<FitResult> {
    run_fit_observed(
        d,
        h,
        src,
        samples,
        eval_weights,
        t,
        output,
        |_| {},
        |_, _| {},
    )
}

/// [`run_fit`], reporting its two phases separately.
///
/// `on_game` fires once per corpus game collected and `on_position` once per
/// position fitted. They are separate because the phases are: collecting the
/// corpus replays games and copies boards, while the fit searches every
/// position twice, and one of the two dominates by orders of magnitude
/// depending on `(d, h)` — a single count over both would extrapolate the
/// first phase's rate onto the second.
///
/// # Errors
///
/// Propagates corpus / I/O errors.
#[allow(clippy::too_many_arguments)]
pub fn run_fit_observed(
    d: u32,
    h: u32,
    src: &Source,
    samples: usize,
    eval_weights: Option<&Path>,
    t: f64,
    output: &Path,
    on_game: impl FnMut(usize),
    on_position: impl FnMut(usize, usize),
) -> Result<FitResult> {
    let positions = collect_positions_observed(src, samples, on_game)?;
    if positions.is_empty() {
        bail!("no positions collected from the corpus");
    }
    let result = match eval_weights {
        Some(p) => {
            let w =
                EvalWeights::load(p).map_err(|e| anyhow::anyhow!("load {}: {e}", p.display()))?;
            fit_with_observed(&positions, d, h, &PatternEval::new(w), on_position)
        }
        None => fit_with_observed(&positions, d, h, &BasicEval::default(), on_position),
    };
    result.to_config(t).save_json(output)?;
    Ok(result)
}

// ===========================================================================
// Multi-ProbCut cascade fit (Phase 6; design doc §4.3.5 / §4.5 B7).
// ===========================================================================

use logistello_search::{DiscPhase, MPC_CASCADE, MpcStageParams, MultiProbCutConfig};

/// One fitted MPC cascade cell: `(disc-phase, h, d)` with its OLS line,
/// residual σ, R² and sample count.
#[derive(Debug, Clone, Copy)]
pub struct MpcCellFit {
    /// Disc-count phase (`Lt36` / `Ge36`, design doc §4.5 B7 split at 36).
    pub phase: DiscPhase,
    /// Search height `h`.
    pub h: u32,
    /// Shallow check depth `d`.
    pub d: u32,
    /// The OLS fit (`a, b, σ, R², n`).
    pub fit: PhaseFit,
}

/// Result of an `--mpc-cascade` fit: every `(phase, h, d)` cell, plus the
/// emitted [`MultiProbCutConfig`].
#[derive(Debug, Clone)]
pub struct McpFitResult {
    /// Per-cell fits, ordered `(h, d)` ascending then `Lt36` before `Ge36`.
    pub cells: Vec<MpcCellFit>,
}

/// Parses an `--mpc-cascade` spec `"h:d1[:d2],..."` into `(h, d1, d2?)`
/// triples. Validates `0 < d1 (< d2) < h` and rejects malformed entries.
///
/// # Errors
///
/// Returns an error on any malformed `h:d1[:d2]` group.
pub fn parse_mpc_cascade(spec: &str) -> Result<Vec<(u32, u32, Option<u32>)>> {
    let mut out = Vec::new();
    for raw in spec.split(',') {
        let g = raw.trim();
        if g.is_empty() {
            continue;
        }
        let parts: Vec<&str> = g.split(':').collect();
        if parts.len() < 2 || parts.len() > 3 {
            bail!("bad --mpc-cascade group '{g}' (want h:d1[:d2])");
        }
        let h: u32 = parts[0]
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("bad height in '{g}'"))?;
        let d1: u32 = parts[1]
            .trim()
            .parse()
            .map_err(|_| anyhow::anyhow!("bad d1 in '{g}'"))?;
        let d2: Option<u32> = match parts.get(2) {
            Some(s) => Some(
                s.trim()
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad d2 in '{g}'"))?,
            ),
            None => None,
        };
        if d1 == 0 || d1 >= h {
            bail!("--mpc-cascade '{g}' needs 0 < d1 < h");
        }
        if let Some(d2) = d2
            && (d2 <= d1 || d2 >= h)
        {
            bail!("--mpc-cascade '{g}' needs d1 < d2 < h");
        }
        out.push((h, d1, d2));
    }
    if out.is_empty() {
        bail!("--mpc-cascade is empty");
    }
    Ok(out)
}

/// Fits every `(phase, h, d)` cascade cell with `eval`.
///
/// For each sampled non-terminal position it computes the deep value `v_h`
/// once per distinct `h` and the shallow value `v_d` once per distinct `d`
/// (all under the production TT discipline, ProbCut/MPC OFF — exactly the
/// true depth-`x` values), stratifies the `(v_d, v_h)` pair into the
/// position's disc-count phase, and runs an independent OLS per
/// `(phase, h, d)` group (design doc §4.5 B7: "per (disc-phase, h, d)
/// linear regression").
/// `(phase_idx, h, d)` → the accumulated `(v_d, v_h)` regression samples.
type CellSamples = std::collections::BTreeMap<(u8, u32, u32), (Vec<f64>, Vec<f64>)>;

/// Fits every `(phase, h, d)` cascade cell with `eval` (see above), calling
/// `on_position` once per corpus position.
///
/// The cascade's cells search to depths that differ by orders of magnitude,
/// but every position pays for *all* of them — the depths are the inner loop —
/// so the outer unit is even and the stage counts positions instead of
/// weighting cells.
fn fit_mpc_with_observed<E: LeafEvaluator>(
    positions: &[GameState],
    cascade: &[(u32, u32, Option<u32>)],
    eval: &E,
    mut on_position: impl FnMut(usize, usize),
) -> McpFitResult {
    use std::collections::BTreeMap;
    // (phase_idx, h, d) -> (xs, ys).
    let mut groups: CellSamples = BTreeMap::new();

    for (i, s) in positions.iter().enumerate() {
        // Ticks before the `continue`: the count is of positions tried.
        on_position(i, positions.len());
        if s.is_terminal() {
            continue;
        }
        let discs = 64 - s.board.empty_count();
        let phase = DiscPhase::for_discs(discs);
        let pidx = match phase {
            DiscPhase::Lt36 => 0u8,
            DiscPhase::Ge36 => 1u8,
        };
        // Distinct depths needed for this position's cascade.
        let mut depths: Vec<u32> = Vec::new();
        for &(h, d1, d2) in cascade {
            for x in [Some(h), Some(d1), d2].into_iter().flatten() {
                if !depths.contains(&x) {
                    depths.push(x);
                }
            }
        }
        // Memoise v_x per distinct depth (so a height shared by several
        // cascade rows is searched once).
        let mut vx: BTreeMap<u32, f64> = BTreeMap::new();
        for &x in &depths {
            vx.insert(x, f64::from(search_value(s, x, eval)));
        }
        for &(h, d1, d2) in cascade {
            let vh = vx[&h];
            for d in std::iter::once(d1).chain(d2) {
                let vd = vx[&d];
                let e = groups.entry((pidx, h, d)).or_default();
                e.0.push(vd);
                e.1.push(vh);
            }
        }
    }

    let mut cells = Vec::new();
    for ((pidx, h, d), (xs, ys)) in groups {
        let phase = if pidx == 0 {
            DiscPhase::Lt36
        } else {
            DiscPhase::Ge36
        };
        cells.push(MpcCellFit {
            phase,
            h,
            d,
            fit: ols(&xs, &ys),
        });
    }
    // Stable, readable ordering: by (h, d) then phase.
    cells.sort_by_key(|c| {
        (
            c.h,
            c.d,
            match c.phase {
                DiscPhase::Lt36 => 0u8,
                DiscPhase::Ge36 => 1u8,
            },
        )
    });
    McpFitResult { cells }
}

impl McpFitResult {
    /// Builds a [`MultiProbCutConfig`] (`enabled = true`, canonical 2-phase
    /// production thresholds 1.0 / 1.4) from this fit.
    #[must_use]
    pub fn to_config(&self) -> MultiProbCutConfig {
        let mut cfg = MultiProbCutConfig::default();
        cfg.enabled = true;
        for c in &self.cells {
            cfg.set_params(
                c.phase,
                c.h,
                c.d,
                MpcStageParams::new(c.fit.a, c.fit.b, c.fit.sigma),
            );
        }
        cfg
    }
}

/// Public entry point for `probcut-fit --mpc-cascade`: collect the corpus,
/// fit every `(phase, h, d)` cascade cell with the chosen evaluator, write
/// the [`MultiProbCutConfig`] JSON, and return the per-cell fits for
/// printing.
///
/// # Errors
///
/// Propagates corpus / I/O errors.
pub fn run_mpc_fit(
    cascade: &[(u32, u32, Option<u32>)],
    src: &Source,
    samples: usize,
    eval_weights: Option<&Path>,
    output: &Path,
) -> Result<McpFitResult> {
    run_mpc_fit_observed(
        cascade,
        src,
        samples,
        eval_weights,
        output,
        |_| {},
        |_, _| {},
    )
}

/// [`run_mpc_fit`], reporting the corpus collection and the cascade fit
/// separately — see [`run_fit_observed`] for why the two are not one count.
///
/// # Errors
///
/// Propagates corpus / I/O errors.
pub fn run_mpc_fit_observed(
    cascade: &[(u32, u32, Option<u32>)],
    src: &Source,
    samples: usize,
    eval_weights: Option<&Path>,
    output: &Path,
    on_game: impl FnMut(usize),
    on_position: impl FnMut(usize, usize),
) -> Result<McpFitResult> {
    let positions = collect_positions_observed(src, samples, on_game)?;
    if positions.is_empty() {
        bail!("no positions collected from the corpus");
    }
    let result = match eval_weights {
        Some(p) => {
            let w =
                EvalWeights::load(p).map_err(|e| anyhow::anyhow!("load {}: {e}", p.display()))?;
            fit_mpc_with_observed(&positions, cascade, &PatternEval::new(w), on_position)
        }
        None => fit_mpc_with_observed(&positions, cascade, &BasicEval::default(), on_position),
    };
    result.to_config().save_json(output)?;
    Ok(result)
}

/// The canonical B7 cascade as an `--mpc-cascade` spec string (design doc
/// §4.5 B7), used as the CLI default and in tests.
#[must_use]
pub fn canonical_mpc_cascade_spec() -> String {
    MPC_CASCADE
        .iter()
        .map(|&(h, d1, d2)| match d2 {
            Some(d2) => format!("{h}:{d1}:{d2}"),
            None => format!("{h}:{d1}"),
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_recovers_exact_line() {
        // y = 2x + 3 exactly: a=2, b=3, sigma=0, R2=1.
        let x: Vec<f64> = (0..50).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 3.0).collect();
        let f = ols(&x, &y);
        assert!((f.a - 2.0).abs() < 1e-9, "a = {}", f.a);
        assert!((f.b - 3.0).abs() < 1e-9, "b = {}", f.b);
        assert!(f.sigma < 1e-6, "sigma = {}", f.sigma);
        assert!((f.r2 - 1.0).abs() < 1e-9, "r2 = {}", f.r2);
        assert_eq!(f.n, 50);
    }

    #[test]
    fn ols_noisy_line_high_r2_positive_sigma() {
        // y = x + small deterministic zig-zag noise: a≈1, R² high, σ>0.
        let x: Vec<f64> = (0..200).map(f64::from).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, v)| v + if i % 2 == 0 { 0.7 } else { -0.7 })
            .collect();
        let f = ols(&x, &y);
        assert!((f.a - 1.0).abs() < 0.05, "a = {}", f.a);
        assert!(f.sigma > 0.0, "sigma must be > 0 with noise");
        assert!(f.r2 > 0.99, "r2 = {} should be high", f.r2);
    }

    #[test]
    fn ols_degenerate_is_unusable() {
        // < 3 points and zero x-variance → a = 0 (unusable → fall through).
        assert_eq!(ols(&[1.0], &[1.0]).a, 0.0);
        let cx = vec![5.0; 20];
        let cy: Vec<f64> = (0..20).map(f64::from).collect();
        assert_eq!(ols(&cx, &cy).a, 0.0, "zero x-variance → a=0");
    }

    #[test]
    fn selfplay_fit_is_sane_and_roundtrips() {
        // Small seeded self-play corpus, BasicEval. a should be near 1
        // (a 2-ply deeper search rarely flips the disc/mobility sign), σ>0,
        // R² strong, both phases produced, JSON loads into ProbCutConfig.
        let dir = std::env::temp_dir();
        let out = dir.join(format!("pcfit_test_{}.json", std::process::id()));
        // Kept small so the unoptimised (debug) `cargo test` stays fast;
        // ~250+ samples/phase is still ample for the OLS sanity bounds.
        // The high-volume fit is exercised by the release CLI demo.
        let res = run_fit(
            4,
            8,
            &Source::Selfplay { games: 20, seed: 1 },
            500,
            None,
            1.5,
            &out,
        )
        .unwrap();

        for ph in [res.lt36, res.ge36] {
            assert!(ph.n >= 50, "each phase has ample samples (n={})", ph.n);
            assert!(ph.a > 0.3 && ph.a < 3.0, "slope a={} in a sane band", ph.a);
            assert!(ph.sigma > 0.0, "sigma must be > 0");
            assert!(
                ph.r2 > 0.5,
                "R² = {} should be strong (BasicEval looser than B7's 0.96 \
                 pattern-eval target — see task report)",
                ph.r2
            );
        }

        let back = ProbCutConfig::load_json(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        assert!(back.enabled, "loaded config is enabled");
        assert_eq!(back.d, 4);
        assert_eq!(back.h, 8);
        assert!(back.params_lt36.is_usable());
        assert!(back.params_ge36.is_usable());
    }

    // ---- Multi-ProbCut cascade fit (Phase 6) ------------------------------

    #[test]
    fn parse_mpc_cascade_canonical_b7() {
        let spec = "3:1,4:2,5:1,6:2,7:3,8:4,9:3:5,10:4:6,11:3:5,12:4,13:5";
        let c = parse_mpc_cascade(spec).unwrap();
        assert_eq!(c.len(), 11);
        assert_eq!(c[0], (3, 1, None));
        assert_eq!(c[6], (9, 3, Some(5)));
        assert_eq!(c[7], (10, 4, Some(6)));
        assert_eq!(c[10], (13, 5, None));
        // Round-trips through the canonical spec helper.
        assert_eq!(parse_mpc_cascade(&canonical_mpc_cascade_spec()).unwrap(), c);
    }

    #[test]
    fn parse_mpc_cascade_rejects_malformed() {
        assert!(parse_mpc_cascade("").is_err(), "empty");
        assert!(parse_mpc_cascade("8").is_err(), "no d");
        assert!(parse_mpc_cascade("8:4:5:6").is_err(), "too many parts");
        assert!(parse_mpc_cascade("8:8").is_err(), "d1 >= h");
        assert!(parse_mpc_cascade("8:0").is_err(), "d1 == 0");
        assert!(parse_mpc_cascade("9:5:3").is_err(), "d2 <= d1");
        assert!(parse_mpc_cascade("9:3:9").is_err(), "d2 >= h");
        assert!(parse_mpc_cascade("x:1").is_err(), "non-numeric");
    }

    #[test]
    fn mpc_cascade_fit_is_sane_and_roundtrips() {
        // Small seeded self-play corpus, BasicEval, a 3-row cascade (keeps
        // the debug `cargo test` fast; the full 11-row fit is the release
        // CLI demo). Every (phase, h, d) cell must be produced with a sane
        // slope, σ>0 and a usable JSON that loads into a MultiProbCutConfig.
        let dir = std::env::temp_dir();
        let out = dir.join(format!("mpcfit_test_{}.json", std::process::id()));
        let cascade = parse_mpc_cascade("8:4,9:3:5,10:4:6").unwrap();
        let res = run_mpc_fit(
            &cascade,
            &Source::Selfplay { games: 24, seed: 2 },
            600,
            None,
            &out,
        )
        .unwrap();

        // Cascade rows: 8→{4}, 9→{3,5}, 10→{4,6} = 5 (h,d) pairs × 2 phases
        // = 10 cells (assuming both phases sampled; corpus is late enough).
        assert!(
            res.cells.len() >= 8,
            "most cells produced ({})",
            res.cells.len()
        );
        // Print the real fitted table (visible with `--nocapture`); this is
        // the canonical `run_mpc_fit` output for a representative cascade.
        eprintln!(
            "[mpc-fit table] cascade=8:4,9:3:5,10:4:6 source=selfplay \
             samples<=600 seed=2 eval=basic cells={}",
            res.cells.len()
        );
        eprintln!("  phase    h   d        a          b      sigma       R2      n");
        for c in &res.cells {
            let ph = match c.phase {
                DiscPhase::Lt36 => "<36 ",
                DiscPhase::Ge36 => ">=36",
            };
            eprintln!(
                "  {ph}  {:>3} {:>3}  {:>9.5} {:>9.5} {:>9.5} {:>8.5} {:>6}",
                c.h, c.d, c.fit.a, c.fit.b, c.fit.sigma, c.fit.r2, c.fit.n
            );
        }
        for c in &res.cells {
            assert!(
                c.fit.n >= 20,
                "cell ({:?},{},{}) n={}",
                c.phase,
                c.h,
                c.d,
                c.fit.n
            );
            assert!(
                c.fit.a > 0.2 && c.fit.a < 3.0,
                "cell ({:?},{},{}) slope a={} sane",
                c.phase,
                c.h,
                c.d,
                c.fit.a
            );
            assert!(c.fit.sigma > 0.0, "σ>0 for ({:?},{},{})", c.phase, c.h, c.d);
            assert!(c.fit.r2 > 0.3, "R²={} usable", c.fit.r2);
        }

        let back = MultiProbCutConfig::load_json(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        assert!(back.enabled, "loaded MPC config is enabled");
        // Every fitted cell is present and usable after the JSON round-trip.
        for c in &res.cells {
            let p = back
                .params_for(c.phase, c.h, c.d)
                .unwrap_or_else(|| panic!("cell ({:?},{},{}) missing", c.phase, c.h, c.d));
            assert!(p.is_usable(), "round-tripped cell usable");
        }
    }

    /// Full canonical B7 cascade fit via the exact production `run_mpc_fit`
    /// code path, printing the complete `(phase, h, d)` table. `#[ignore]`d
    /// because a depth-13 fit over a real corpus is expensive; run
    /// explicitly with `--ignored --nocapture` to produce the demo table.
    /// Same command path as the CLI `probcut-fit --mpc-cascade`, just driven
    /// from a test so the table is captured deterministically.
    #[test]
    #[ignore]
    fn demo_full_b7_cascade_fit_table() {
        let dir = std::env::temp_dir();
        let out = dir.join(format!("mpc_demo_{}.json", std::process::id()));
        let cascade = parse_mpc_cascade(&canonical_mpc_cascade_spec()).unwrap();
        assert_eq!(cascade.len(), 11, "full B7 cascade");
        // Seed 2 matches the task's demo invocation. The full B7 cascade
        // fit recomputes 13 distinct NegaScout depths (1..=13, incl. a full
        // depth-13 minimax) per position, so it is *very* expensive; the
        // sample count here is kept small enough to complete in a tractable
        // window while every (phase, h, d) cell still has a non-trivial n
        // (the high-volume production fit is the CLI `probcut-fit
        // --mpc-cascade --samples N` path; this just captures a real,
        // deterministic table from the identical `run_mpc_fit` code).
        let samples = 220;
        let res = run_mpc_fit(
            &cascade,
            &Source::Selfplay {
                games: samples,
                seed: 2,
            },
            samples,
            None,
            &out,
        )
        .unwrap();
        eprintln!(
            "probcut-fit mpc-cascade(canonical B7) source=selfplay samples<={samples} \
             seed=2 eval=basic T(prod)=1.0/1.4 cells={}",
            res.cells.len()
        );
        eprintln!("  phase    h   d        a          b      sigma       R2      n");
        for c in &res.cells {
            let ph = match c.phase {
                DiscPhase::Lt36 => "<36 ",
                DiscPhase::Ge36 => ">=36",
            };
            eprintln!(
                "  {ph}  {:>3} {:>3}  {:>9.5} {:>9.5} {:>9.5} {:>8.5} {:>6}",
                c.h, c.d, c.fit.a, c.fit.b, c.fit.sigma, c.fit.r2, c.fit.n
            );
        }
        let back = MultiProbCutConfig::load_json(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        assert!(back.enabled);
        // All 11 cascade rows → 13 distinct (h,d) pairs; with a small,
        // early-game-heavy corpus the `>=36` phase may be sparse, so we
        // only require every produced cell to be a usable round-tripped
        // fit (per-cell statistical sanity is the dedicated unit test's
        // job; this test's purpose is the printed demo table).
        assert!(res.cells.len() >= 13, "cells={}", res.cells.len());
        for c in &res.cells {
            assert!(
                back.params_for(c.phase, c.h, c.d).is_some(),
                "cell ({:?},{},{}) round-trips",
                c.phase,
                c.h,
                c.d
            );
        }
    }
}
