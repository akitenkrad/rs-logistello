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

/// Collects up to `max_samples` non-terminal positions from the corpus.
///
/// Self-play reuses the exact deterministic seeding of the Phase-4b
/// extractor; WTHOR replays each `.wtb` game (passes auto-inserted by the
/// reader, design doc §4.5 B6). Positions are taken in game order.
fn collect_positions(src: &Source, max_samples: usize) -> Result<Vec<GameState>> {
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

/// Runs the single-pair `(d, h)` fit over the corpus with `eval`.
fn fit_with<E: LeafEvaluator>(positions: &[GameState], d: u32, h: u32, eval: &E) -> FitResult {
    let mut lt_x = Vec::new();
    let mut lt_y = Vec::new();
    let mut ge_x = Vec::new();
    let mut ge_y = Vec::new();
    for s in positions {
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
    let positions = collect_positions(src, samples)?;
    if positions.is_empty() {
        bail!("no positions collected from the corpus");
    }
    let result = match eval_weights {
        Some(p) => {
            let w =
                EvalWeights::load(p).map_err(|e| anyhow::anyhow!("load {}: {e}", p.display()))?;
            fit_with(&positions, d, h, &PatternEval::new(w))
        }
        None => fit_with(&positions, d, h, &BasicEval::default()),
    };
    result.to_config(t).save_json(output)?;
    Ok(result)
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
}
