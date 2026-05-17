//! Drawishness ("引き分けやすさ") move-preference adjustment
//! (design doc §4.3.7 / Buro 1999, "Toward Opening Book Learning").
//!
//! # What the design doc / Buro 1999 ask for
//!
//! Design doc §4.3.7:
//!
//! > drawishness 関数は **「相手がミスしやすい局面」** を選好する補正項であり，
//! > 完璧な評価とは別に勝率最大化の観点で局面を評価する.
//!
//! Buro (1999, "Toward Opening Book Learning", *ICCA Journal* 22(2):98-102)
//! observes that a *Negamax-optimal* book always steers the program into the
//! single value-maximising line, which (a) makes the program predictable and
//! (b) ignores that a slightly sub-optimal line where the **opponent** has
//! many tempting-but-losing replies wins more games in practice against a
//! fallible opponent. He therefore blends the pure book value with a term
//! that rewards positions in which the opponent is more likely to err.
//!
//! # Concrete `trap` term used here (documented, simple, bounded)
//!
//! The book stores, per position, the negamax value of every in-book child
//! move (see [`learner`](crate::learner)). For the side **to move at the
//! parent**, picking child `c` hands the opponent the position `c` with its
//! own set of in-book continuations (the grand-children). The opponent then
//! chooses among those grand-child values. We define the *trap* of a child
//! `c` as the **spread of the opponent's continuation values at `c`**:
//!
//! ```text
//! trap(c) = max_g value(g)  -  min_g value(g)        (g ranges over c's
//!                                                     in-book children)
//! ```
//!
//! measured from the **opponent's** point of view at `c` (i.e. the same
//! `-child.value` quantities the parent's negamax takes the max of, one ply
//! deeper). A large spread means the opponent has both strong and weak
//! replies available — an easy position to *err* in (pick a weak one). A
//! position with a single forced reply, or where every reply is equal, has
//! `trap = 0`: there is nothing to trap the opponent into. This is the
//! "many tempting but losing continuations for the opponent" reading of
//! Buro 1999, expressed with the data the book already has and nothing more
//! (we deliberately do **not** invent a richer scheme — §4.3.7 only asks for
//! a faithful, documented preference shift).
//!
//! `trap(c)` is `0` when `c` has fewer than two in-book children (no spread
//! is defined) and is otherwise a non-negative integer bounded by the
//! evaluation scale (`<= 2 * SCORE_BOUND`), so the blended score below is
//! always finite and bounded.
//!
//! # The blended preference score
//!
//! For a candidate move `m` leading to child value `v(m)` (the parent-POV
//! negamax value `-child.value`) and trap term `t(m)`:
//!
//! ```text
//! score(m, λ) = (1 - λ) · v(m)  +  λ · t(m),      λ ∈ [0, 0.5]
//! ```
//!
//! - `λ = 0` ⇒ `score = v(m)` exactly: the booked move is the pure
//!   Negamax-optimal one (this is asserted by the gold test — drawishness at
//!   `λ = 0` must be a no-op).
//! - increasing `λ` monotonically increases the weight of `t(m)`, so a move
//!   can only ever be *promoted* over the negamax-best one by having a
//!   strictly larger trap term; the effect is monotone in `λ`.
//! - both terms are bounded, so `score` is bounded.
//!
//! `λ` is clamped into `[0, 0.5]` (design doc §6: `--drawishness` range
//! `0.0 .. 0.5`). The function is intentionally tiny and pure so it can be
//! unit-tested on hand-constructed sibling sets in isolation.

/// Evaluation-scale bound (disc scale): leaf / negamax values stay well
/// inside `±(64 + 64)` (design doc §4.5 B6). The trap spread is therefore
/// bounded by `2 * SCORE_BOUND`, keeping [`score`] finite and bounded.
pub const SCORE_BOUND: i32 = 128;

/// Clamps a raw `--drawishness` parameter into the design-doc §6 range
/// `[0.0, 0.5]` (and maps NaN to `0.0`, the pure-Negamax default).
#[must_use]
pub fn clamp_lambda(lambda: f64) -> f64 {
    if lambda.is_nan() {
        0.0
    } else {
        lambda.clamp(0.0, 0.5)
    }
}

/// The *trap* term of a candidate child given the **opponent's** in-book
/// continuation values at that child (one ply deeper, opponent POV).
///
/// `opp_child_values` is the list of negamax values, from the *opponent's*
/// point of view, of every in-book reply the opponent has at the child.
///
/// Returns the spread `max - min`, or `0` when fewer than two replies are
/// known (no spread is defined — nothing to trap into). Always `>= 0` and
/// `<= 2 * SCORE_BOUND` for in-range inputs (design doc §4.3.7 / Buro 1999).
#[must_use]
pub fn trap(opp_child_values: &[i32]) -> i32 {
    if opp_child_values.len() < 2 {
        return 0;
    }
    let mut lo = opp_child_values[0];
    let mut hi = opp_child_values[0];
    for &v in &opp_child_values[1..] {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    hi - lo
}

/// The blended drawishness preference score of a candidate move
/// (design doc §4.3.7 / Buro 1999):
///
/// ```text
/// score(m, λ) = (1 - λ) · negamax_value  +  λ · trap
/// ```
///
/// - `negamax_value` is the parent-POV negamax value of the child (`= the
///   `-child.value` quantity the parent's negamax takes the `max` of).
/// - `trap` is [`trap`] of the child (opponent-error potential, `>= 0`).
/// - `lambda` is clamped to `[0, 0.5]` via [`clamp_lambda`].
///
/// At `λ = 0` this is exactly `negamax_value` (pure Negamax-optimal). The
/// result is finite and bounded whenever the inputs are in the evaluation
/// scale.
#[must_use]
pub fn score(negamax_value: i32, trap: i32, lambda: f64) -> f64 {
    let l = clamp_lambda(lambda);
    (1.0 - l) * f64::from(negamax_value) + l * f64::from(trap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_lambda_bounds_and_nan() {
        assert_eq!(clamp_lambda(-1.0), 0.0);
        assert_eq!(clamp_lambda(0.0), 0.0);
        assert_eq!(clamp_lambda(0.3), 0.3);
        assert_eq!(clamp_lambda(0.5), 0.5);
        assert_eq!(clamp_lambda(0.9), 0.5);
        assert_eq!(clamp_lambda(f64::NAN), 0.0);
    }

    #[test]
    fn trap_zero_for_fewer_than_two_children() {
        assert_eq!(trap(&[]), 0);
        assert_eq!(trap(&[7]), 0);
    }

    #[test]
    fn trap_is_spread_and_nonnegative() {
        assert_eq!(trap(&[3, 3, 3]), 0); // all equal -> nothing to trap into
        assert_eq!(trap(&[-10, 5]), 15);
        assert_eq!(trap(&[5, -10, 2, 9, -3]), 19);
        // order-independent
        assert_eq!(trap(&[9, -3, 5, 2, -10]), 19);
    }

    #[test]
    fn trap_bounded_by_scale() {
        let t = trap(&[-SCORE_BOUND, SCORE_BOUND]);
        assert!((0..=2 * SCORE_BOUND).contains(&t));
        assert_eq!(t, 2 * SCORE_BOUND);
    }

    #[test]
    fn score_at_lambda_zero_is_pure_negamax() {
        // λ = 0 must ignore the trap term entirely (gold invariant: a
        // λ=0 book is exactly the Negamax-optimal book).
        for &v in &[-37, -1, 0, 4, 51] {
            for &t in &[0, 1, 99, 256] {
                assert_eq!(score(v, t, 0.0), f64::from(v));
            }
        }
    }

    #[test]
    fn score_monotone_nondecreasing_in_lambda_when_trap_exceeds_value() {
        // If trap > value, raising λ strictly increases the score; the
        // ranking effect is monotone in λ.
        let v = 10;
        let t = 40;
        let s0 = score(v, t, 0.0);
        let s1 = score(v, t, 0.2);
        let s2 = score(v, t, 0.4);
        let s3 = score(v, t, 0.5);
        assert!(s0 < s1 && s1 < s2 && s2 < s3, "{s0} {s1} {s2} {s3}");
    }

    #[test]
    fn score_promotes_only_toward_higher_trap() {
        // Two siblings: A is negamax-better, B has a strictly larger trap.
        // At λ=0 A wins; B can only overtake A as λ grows (toward higher
        // trap), never the reverse.
        let (va, ta) = (12, 2);
        let (vb, tb) = (9, 60);
        assert!(score(va, ta, 0.0) > score(vb, tb, 0.0)); // λ=0: negamax A
        // At a large enough λ the higher-trap B overtakes.
        assert!(score(vb, tb, 0.5) > score(va, ta, 0.5));
        // The crossover is monotone: once B leads it keeps leading.
        let crossed: Vec<bool> = (0..=50)
            .map(|i| {
                let l = f64::from(i) / 100.0; // 0.00 .. 0.50
                score(vb, tb, l) > score(va, ta, l)
            })
            .collect();
        // monotone step function false...false,true...true
        let first_true = crossed.iter().position(|&b| b);
        if let Some(k) = first_true {
            assert!(crossed[k..].iter().all(|&b| b), "ranking flips back");
        }
    }

    #[test]
    fn score_is_finite_and_bounded() {
        for &v in &[-SCORE_BOUND, 0, SCORE_BOUND] {
            for &t in &[0, 2 * SCORE_BOUND] {
                for &l in &[0.0, 0.25, 0.5] {
                    let s = score(v, t, l);
                    assert!(s.is_finite());
                    assert!(s.abs() <= 2.0 * f64::from(SCORE_BOUND));
                }
            }
        }
    }
}
