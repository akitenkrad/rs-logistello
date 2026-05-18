//! `eval-correlation-edax`: Pearson correlation of our `PatternEval` value
//! against Edax's `hint`/`go` evaluation, on a bounded set of positions
//! (design doc `Logistello.md` §4.3.8 `eval_correlation_edax`, "Edax as
//! ground truth"; Phase 9b).
//!
//! The Pearson coefficient itself is pure and unit-tested here; the
//! position generation + Edax querying lives in `main.rs` (it needs the
//! real engine / GTP session and is therefore bounded & skip-if-absent).

use anyhow::{Context, Result};

/// Pearson product-moment correlation `r` of paired samples `(xs, ys)`.
///
/// Returns `None` if `n < 2` or either series has zero variance (`r`
/// undefined). `r ∈ [-1, 1]`.
#[must_use]
pub fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    let n = xs.len();
    if n != ys.len() || n < 2 {
        return None;
    }
    let nf = n as f64;
    let mx = xs.iter().sum::<f64>() / nf;
    let my = ys.iter().sum::<f64>() / nf;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mx;
        let dy = y - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None;
    }
    Some(sxy / (sxx.sqrt() * syy.sqrt()))
}

/// One position's paired (our score, Edax score) sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrSample {
    /// 0-based sample index (stable order for the CSV).
    pub idx: usize,
    /// Our `PatternEval` leaf value (side-to-move POV, disc scale).
    pub our_score: i32,
    /// Edax's reported evaluation at the same position (disc scale).
    pub edax_score: i32,
}

/// Writes the samples + the Pearson `r` as a deterministic CSV.
///
/// # Errors
///
/// Propagates I/O errors.
pub fn write_csv(path: &std::path::Path, samples: &[CorrSample], r: Option<f64>) -> Result<()> {
    use std::io::Write;
    let mut buf = String::from("idx,our_score,edax_score\n");
    for s in samples {
        buf.push_str(&format!("{},{},{}\n", s.idx, s.our_score, s.edax_score));
    }
    buf.push_str(&format!(
        "# pearson_r={} n={}\n",
        r.map_or_else(|| "NA".to_string(), |v| format!("{v:.6}")),
        samples.len()
    ));
    let mut f =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(buf.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_perfect_positive_and_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let yp = [2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let yn = [10.0, 8.0, 6.0, 4.0, 2.0]; // y = -2x + c
        assert!((pearson(&x, &yp).unwrap() - 1.0).abs() < 1e-12);
        assert!((pearson(&x, &yn).unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn pearson_known_value() {
        // Classic textbook pair: r ≈ 0.9758 (Pearson example).
        let x = [43.0, 21.0, 25.0, 42.0, 57.0, 59.0];
        let y = [99.0, 65.0, 79.0, 75.0, 87.0, 81.0];
        let r = pearson(&x, &y).unwrap();
        assert!((r - 0.5298).abs() < 1e-3, "r={r}");
    }

    #[test]
    fn pearson_guards_degenerate_inputs() {
        assert!(pearson(&[], &[]).is_none());
        assert!(pearson(&[1.0], &[2.0]).is_none(), "n<2");
        assert!(pearson(&[1.0, 2.0], &[3.0]).is_none(), "len mismatch");
        // zero variance in x
        assert!(pearson(&[5.0, 5.0, 5.0], &[1.0, 2.0, 3.0]).is_none());
        // zero variance in y
        assert!(pearson(&[1.0, 2.0, 3.0], &[7.0, 7.0, 7.0]).is_none());
    }

    #[test]
    fn pearson_is_in_range() {
        let x = [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 13.0];
        let y = [2.0, 1.0, 4.0, 3.0, 7.0, 5.0, 20.0];
        let r = pearson(&x, &y).unwrap();
        assert!((-1.0..=1.0).contains(&r), "r out of range: {r}");
    }

    #[test]
    fn csv_round_and_deterministic() {
        let s = vec![
            CorrSample {
                idx: 0,
                our_score: 3,
                edax_score: 5,
            },
            CorrSample {
                idx: 1,
                our_score: -2,
                edax_score: -1,
            },
        ];
        let r = pearson(&[3.0, -2.0], &[5.0, -1.0]);
        let dir = std::env::temp_dir();
        let p = dir.join(format!("ec_{}.csv", std::process::id()));
        write_csv(&p, &s, r).unwrap();
        let txt = std::fs::read_to_string(&p).unwrap();
        let _ = std::fs::remove_file(&p);
        assert!(txt.starts_with("idx,our_score,edax_score\n"));
        assert!(txt.contains("# pearson_r="));
    }
}
