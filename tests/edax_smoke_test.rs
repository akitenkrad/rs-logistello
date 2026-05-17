//! Workspace integration smoke test for the Phase 9a Edax v4.6 wiring
//! (design doc `Logistello.md` §4.5 **B5**, "Edax integration —
//! canonical").
//!
//! What this verifies:
//! 1. The shared [`logistello_cli::edax::EdaxConfig`] helper builds the
//!    correct **GTP**, book-off, single-thread (deterministic)
//!    invocation, and an [`ExternalEnginePlayer`] can be constructed from
//!    it (pure-logic; runs everywhere).
//! 2. When the gitignored `.edax/` install is present, the **real Edax
//!    binary + eval weights** answer a direct GTP handshake
//!    (`boardsize`/`clear_board`/`genmove`) with a **legal move** from the
//!    standard opening, and do so **deterministically**.
//!
//! **Skip-when-absent contract:** `.edax/` is gitignored (Edax is a
//! platform binary + multi-MB weights, never committed). When
//! `.edax/edax` is missing the engine part prints a skip notice and the
//! test **passes**, so CI and fresh clones stay green.
//!
//! Why the real-engine part uses a *direct* GTP exchange rather than
//! `ExternalEnginePlayer::select_move`: the pinned `rs-othello-sim`
//! `GtpProtocol` re-notifies the engine of its own move after `genmove`
//! (`play <own_color> <own_move>`), which standard GTP engines like real
//! Edax reject (`? wrong color`, since `genmove` already committed the
//! move). That rs-othello-sim limitation is documented in
//! `crates/logistello-cli/src/edax.rs`; it does not affect the
//! correctness of the binary, weights, or the GTP invocation produced
//! here (which Phase 9b drives via the direct-GTP discipline).

use std::time::Duration;

use logistello_cli::edax::{EDAX_PROTOCOL, EdaxConfig, locate_default_install};
use othello_core::{Color, GameState};
use othello_player::{ExternalEnginePlayer, Protocol};

/// Pure-logic: the helper builds a GTP / book-off / single-thread config,
/// and an `ExternalEnginePlayer` constructs from it. Runs everywhere.
#[test]
fn edax_config_is_gtp_book_off_and_constructs_a_player() {
    let cfg = EdaxConfig::new(EdaxConfig::DEFAULT_EDAX_PATH, 6);
    assert_eq!(EDAX_PROTOCOL, Protocol::Gtp);

    let ec = cfg.external_engine_config();
    assert_eq!(ec.protocol, Protocol::Gtp);

    let args = cfg.args();
    assert_eq!(args[0], "-gtp", "Edax must be launched in GTP mode");
    assert!(
        args.windows(2)
            .any(|w| w[0] == "-book-usage" && w[1] == "off"),
        "the opening book must be disabled (deterministic; none installed)"
    );
    assert!(
        args.windows(2).any(|w| w[0] == "-n" && w[1] == "1"),
        "single-threaded -> deterministic"
    );

    // Construct the rs-othello-sim player (Phase 9a deliverable: the
    // wiring is reachable). We do not call select_move here — see the
    // module docs / edax.rs for the rs-othello-sim self-notify caveat.
    let _player = ExternalEnginePlayer::new(Color::Black, cfg.external_engine_config());
}

/// Real engine: a direct GTP handshake against the locally built Edax
/// returns a **legal** first move from the standard opening, twice
/// (deterministic). Skips (passing) when `.edax/` is absent.
#[test]
fn edax_real_engine_returns_a_legal_deterministic_move() {
    let Some(binary) = locate_default_install() else {
        eprintln!(
            "SKIP: no .edax/edax found up the tree (gitignored Edax install \
             absent). Run scripts/setup_edax.sh to enable this test. Passing."
        );
        return;
    };

    // Low level keeps the smoke fast; generous timeout for CI safety.
    let cfg = EdaxConfig::new(&binary, 4).with_timeout(Duration::from_secs(60));

    if !cfg.is_available() {
        eprintln!("SKIP: {:?} not a file. Passing.", cfg.binary);
        return;
    }
    if !cfg.eval_file.is_file() {
        eprintln!(
            "SKIP: Edax binary present but eval weights {:?} missing. \
             Re-run scripts/setup_edax.sh. Passing.",
            cfg.eval_file
        );
        return;
    }

    eprintln!(
        "Edax found at {:?} (level {}); driving via {:?} (direct GTP).",
        cfg.binary, cfg.level, EDAX_PROTOCOL
    );

    // Set of legal Black moves from the standard opening (algebraic,
    // lowercase): d3, c4, f5, e6.
    let start = GameState::standard_8x8();
    let legal: Vec<String> = start
        .legal_moves()
        .iter()
        .filter_map(|m| match m {
            othello_core::Move::Place(c) => {
                Some(format!("{}{}", (b'a' + c.col) as char, c.row + 1))
            }
            othello_core::Move::Pass => None,
        })
        .collect();
    assert_eq!(start.legal_moves().len(), 4, "standard opening has 4 moves");

    let mv1 = cfg
        .genmove_from_start()
        .expect("Edax must answer the direct GTP handshake (run 1)");
    let mv2 = cfg
        .genmove_from_start()
        .expect("Edax must answer the direct GTP handshake (run 2)");

    eprintln!("Edax genmove (run 1) = {mv1:?}, (run 2) = {mv2:?}; legal = {legal:?}");

    assert!(
        legal.iter().any(|m| m.eq_ignore_ascii_case(&mv1)),
        "Edax returned an ILLEGAL move {mv1:?}; legal from standard = {legal:?}"
    );
    assert_eq!(
        mv1, mv2,
        "Edax with -level/-book-usage off/-n 1 must be deterministic"
    );
}
