//! Shared Edax v4.6 external-engine configuration (design doc
//! `Logistello.md` §4.5 **B5**, "Edax integration — canonical"; Phase 9a).
//!
//! Edax is the fixed-strength external opponent and non-playing oracle.
//! This module owns the single place that knows *how to drive the locally
//! built Edax binary*, so Phase 9b's `elo-vs-edax` only has to call
//! [`EdaxConfig::external_engine_config`].
//!
//! ## Why GTP (not Ntest)?
//!
//! `rs-othello-sim`'s [`ExternalEnginePlayer`] speaks two dialects
//! ([`Protocol::Gtp`], [`Protocol::Ntest`]). Edax is launched with `-gtp`
//! and then speaks **standard GTP** (`src/gtp.c`): `boardsize 8` /
//! `clear_board` / `play <c> <m>` / `genmove <c>` with `= …\n\n`
//! responses — exactly the wire format `rs-othello-sim`'s GTP parser
//! expects. The `Ntest` dialect uses `set game <board>` + `go`, which Edax
//! does **not** understand: Edax's text UI uses `setboard` (not
//! `set game`), silently ignores the unknown `set game` line, and then
//! `go` plays from the *default* opening — the supplied position is lost.
//! GTP is therefore the correct and only viable dialect.
//!
//! ## Known limitation: rs-othello-sim self-notify vs. standard GTP
//!
//! `rs-othello-sim`'s `GtpProtocol` (the frozen, pinned reused library —
//! rev `0ba79a91…`, not modifiable here) re-notifies the engine of *its
//! own* move after `genmove` (`select_move` calls
//! `notify_move(self.color, mv)` → `play <own_color> <own_move>`). That is
//! correct for the mock GTP scripts it was tested against, but **wrong for
//! standard GTP engines**: real Edax already *commits* the move during
//! `genmove` (advancing the side to move), so the redundant
//! `play <own_color> <own_move>` is rejected with `? wrong color`, which
//! surfaces as a `PlayerError` out of the very first `select_move`.
//!
//! Consequence: driving real Edax through `ExternalEnginePlayer` end-to-end
//! is not possible with this pinned `rs-othello-sim` revision. Phase 9b's
//! `elo-vs-edax` must therefore drive Edax via a *direct* GTP exchange
//! (the wire protocol [`EdaxConfig::args`] selects), not via
//! `ExternalEnginePlayer::select_move`. The binary, weights and GTP
//! invocation produced here are correct and verified (see
//! `tests/edax_smoke_test.rs`, which performs a direct
//! `boardsize`/`clear_board`/`genmove` handshake against the real engine).
//!
//! ## Deterministic fixed-strength invocation
//!
//! ```text
//! .edax/edax -gtp -eval-file <ABS>/.edax/data/eval.dat \
//!   -level <N> -book-usage off -n 1 -q
//! ```
//!
//! - `-level N` — search level (fixed strength).
//! - `-book-usage off` — disable the opening book (none is installed).
//! - `-n 1` — single thread → deterministic.
//! - `-q` — quiet (no board redisplay; clean GTP stream).
//! - `-eval-file` is an **absolute** path so the working directory is
//!   irrelevant.
//!
//! See `EDAX_SETUP.md` (repo root) and `scripts/setup_edax.sh` for how the
//! gitignored `.edax/` directory is regenerated.

use std::path::{Path, PathBuf};
use std::time::Duration;

use othello_core::BoardSize;
use othello_player::{ExternalEngineConfig, Protocol};

/// How Edax is driven. See the module docs for the GTP-vs-Ntest rationale.
pub const EDAX_PROTOCOL: Protocol = Protocol::Gtp;

/// Configuration describing the locally built Edax engine.
///
/// Build an [`ExternalEngineConfig`] (the type `rs-othello-sim`'s
/// [`othello_player::ExternalEnginePlayer`] consumes) with
/// [`EdaxConfig::external_engine_config`].
#[derive(Debug, Clone)]
pub struct EdaxConfig {
    /// Path to the Edax binary (default: [`EdaxConfig::DEFAULT_EDAX_PATH`]).
    pub binary: PathBuf,
    /// Path to the evaluation weights. Defaults to
    /// `<binary-dir>/data/eval.dat` when [`EdaxConfig::new`] is used.
    pub eval_file: PathBuf,
    /// Fixed search strength (`edax -level N`).
    pub level: u32,
    /// Per-move timeout for the external-engine player.
    pub timeout: Duration,
}

impl EdaxConfig {
    /// Default Edax binary path, relative to the repo root. Phase 9b's
    /// `--edax-path` flag defaults to this.
    pub const DEFAULT_EDAX_PATH: &'static str = ".edax/edax";

    /// Default evaluation-weights path, relative to the repo root.
    pub const DEFAULT_EVAL_PATH: &'static str = ".edax/data/eval.dat";

    /// Default fixed strength used for Elo measurement.
    pub const DEFAULT_LEVEL: u32 = 10;

    /// Builds a config for `binary` at the given `level`, deriving the
    /// eval-weights path as `<binary-dir>/data/eval.dat` (the layout
    /// produced by `scripts/setup_edax.sh`).
    #[must_use]
    pub fn new(binary: impl Into<PathBuf>, level: u32) -> Self {
        let binary = binary.into();
        let eval_file = binary.parent().map_or_else(
            || PathBuf::from("data/eval.dat"),
            |d| d.join("data/eval.dat"),
        );
        Self {
            binary,
            eval_file,
            level,
            timeout: Duration::from_secs(30),
        }
    }

    /// Config for the default gitignored install (`.edax/edax`,
    /// `.edax/data/eval.dat`) at the default level.
    #[must_use]
    pub fn default_install() -> Self {
        Self::new(Self::DEFAULT_EDAX_PATH, Self::DEFAULT_LEVEL)
    }

    /// Overrides the per-move timeout (builder).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Overrides the eval-weights path (builder).
    #[must_use]
    pub fn with_eval_file(mut self, eval_file: impl Into<PathBuf>) -> Self {
        self.eval_file = eval_file.into();
        self
    }

    /// Returns `true` iff the configured Edax binary exists on disk.
    ///
    /// Callers (tests, Phase 9b) use this to **skip gracefully** when the
    /// gitignored `.edax/` install is absent rather than failing — Edax is
    /// never committed, so CI / fresh clones must stay green.
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.binary.is_file()
    }

    /// The exact Edax CLI arguments (the deterministic fixed-strength,
    /// book-off, GTP invocation). The eval-file path is canonicalised to
    /// an absolute path so the spawned process's working directory does
    /// not matter.
    #[must_use]
    pub fn args(&self) -> Vec<String> {
        let eval =
            std::fs::canonicalize(&self.eval_file).unwrap_or_else(|_| self.eval_file.clone());
        vec![
            "-gtp".to_string(),
            "-eval-file".to_string(),
            eval.to_string_lossy().into_owned(),
            "-level".to_string(),
            self.level.to_string(),
            "-book-usage".to_string(),
            "off".to_string(),
            "-n".to_string(),
            "1".to_string(),
            "-q".to_string(),
        ]
    }

    /// Builds the `rs-othello-sim` [`ExternalEngineConfig`] for an
    /// [`othello_player::ExternalEnginePlayer`] driving this Edax.
    ///
    /// NOTE: see the module-level "Known limitation" — the pinned
    /// `rs-othello-sim` `GtpProtocol` re-notifies the engine of its own
    /// move after `genmove`, which real Edax rejects (`? wrong color`).
    /// This constructor is still provided (Phase 9b may use a future
    /// rs-othello-sim that fixes this, and the config itself is correct),
    /// but reliable single-move queries should use
    /// [`EdaxConfig::genmove_from_start`] / a direct GTP exchange.
    #[must_use]
    pub fn external_engine_config(&self) -> ExternalEngineConfig {
        let mut cfg =
            ExternalEngineConfig::new(self.binary.clone(), EDAX_PROTOCOL, BoardSize::STANDARD);
        cfg.args = self.args();
        cfg.timeout = self.timeout;
        cfg
    }

    /// Direct, blocking GTP handshake: spawn Edax with [`Self::args`],
    /// send `boardsize 8` / `clear_board` / `genmove black`, and return
    /// Edax's chosen first move as a lowercase GTP coordinate (e.g.
    /// `"f5"`), or `"pass"`.
    ///
    /// This is the *reliable* way to query real Edax (it does not use the
    /// pinned `rs-othello-sim` self-notify-after-`genmove` path that Edax
    /// rejects). Phase 9b's `elo-vs-edax` drives Edax over a long-lived
    /// session built on this same direct-GTP discipline (alternating
    /// `play`/`genmove`, never re-`play`ing Edax's own move).
    ///
    /// Returns `Err` if the binary is missing, the process cannot be
    /// spawned, or Edax does not answer within `self.timeout`.
    pub fn genmove_from_start(&self) -> std::io::Result<String> {
        use std::io::{BufRead, BufReader, Write};
        use std::process::{Command, Stdio};
        use std::sync::mpsc;
        use std::thread;

        let mut child = Command::new(&self.binary)
            .args(self.args())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("no stdout"))?;

        // Reader thread: parse GTP responses (`= body\n\n` / `? err\n\n`),
        // forward each response body over a channel.
        let (tx, rx) = mpsc::channel::<Result<String, String>>();
        let reader = thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            let mut first: Option<String> = None;
            loop {
                let mut line = String::new();
                match r.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {}
                    Err(_) => break,
                }
                let t = line.trim_end_matches(['\r', '\n']).to_string();
                if t.is_empty() {
                    if let Some(h) = first.take() {
                        let msg = if let Some(b) = h.strip_prefix('=') {
                            Ok(b.trim().to_string())
                        } else if let Some(e) = h.strip_prefix('?') {
                            Err(e.trim().to_string())
                        } else {
                            Err(format!("unexpected GTP line: {h:?}"))
                        };
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                } else if first.is_none() {
                    first = Some(t);
                }
            }
        });

        let recv = |rx: &mpsc::Receiver<Result<String, String>>| -> std::io::Result<String> {
            rx.recv_timeout(self.timeout)
                .map_err(|_| std::io::Error::other("Edax GTP response timed out"))?
                .map_err(std::io::Error::other)
        };

        writeln!(stdin, "boardsize 8")?;
        stdin.flush()?;
        recv(&rx)?;
        writeln!(stdin, "clear_board")?;
        stdin.flush()?;
        recv(&rx)?;
        writeln!(stdin, "genmove black")?;
        stdin.flush()?;
        let mv = recv(&rx)?;

        let _ = writeln!(stdin, "quit");
        let _ = stdin.flush();
        drop(stdin);
        let _ = child.wait();
        let _ = reader.join();

        Ok(mv.to_ascii_lowercase())
    }
}

/// Resolves the Edax binary path: the CLI-provided `--edax-path` if given,
/// otherwise [`EdaxConfig::DEFAULT_EDAX_PATH`].
#[must_use]
pub fn resolve_edax_path(cli_path: Option<&Path>) -> PathBuf {
    cli_path.map_or_else(
        || PathBuf::from(EdaxConfig::DEFAULT_EDAX_PATH),
        Path::to_path_buf,
    )
}

/// Best-effort absolute path to the gitignored `.edax/` install,
/// independent of the current working directory.
///
/// The default `.edax/edax` path is *relative to the repo root*, but the
/// process cwd is not guaranteed to be the repo root (e.g. `cargo test`
/// runs with the crate directory as cwd). This walks up from
/// `CARGO_MANIFEST_DIR` (and then the cwd) looking for a directory that
/// contains `.edax/edax`, returning that absolute binary path if found.
///
/// Returns `None` if no `.edax/edax` is found anywhere up the tree (the
/// expected case on CI / fresh clones — callers then skip gracefully).
#[must_use]
pub fn locate_default_install() -> Option<PathBuf> {
    fn search_from(start: &Path) -> Option<PathBuf> {
        let mut dir = Some(start);
        while let Some(d) = dir {
            let cand = d.join(EdaxConfig::DEFAULT_EDAX_PATH);
            if cand.is_file() {
                return Some(cand);
            }
            dir = d.parent();
        }
        None
    }

    // 1. Walk up from this crate's manifest dir (stable under `cargo test`).
    if let Some(p) = option_env!("CARGO_MANIFEST_DIR")
        .map(Path::new)
        .and_then(search_from)
    {
        return Some(p);
    }
    // 2. Fall back to walking up from the current working directory.
    std::env::current_dir()
        .ok()
        .as_deref()
        .and_then(search_from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_paths() {
        assert_eq!(EdaxConfig::DEFAULT_EDAX_PATH, ".edax/edax");
        assert_eq!(EdaxConfig::DEFAULT_EVAL_PATH, ".edax/data/eval.dat");
    }

    #[test]
    fn eval_file_derived_from_binary_dir() {
        let c = EdaxConfig::new(".edax/edax", 8);
        assert_eq!(c.eval_file, PathBuf::from(".edax/data/eval.dat"));
        assert_eq!(c.level, 8);
    }

    #[test]
    fn args_are_deterministic_book_off_gtp() {
        let c = EdaxConfig::new("/nonexistent/edax", 12);
        let a = c.args();
        assert_eq!(a[0], "-gtp");
        assert!(a.iter().any(|s| s == "-book-usage"));
        assert!(a.windows(2).any(|w| w[0] == "-book-usage" && w[1] == "off"));
        assert!(a.windows(2).any(|w| w[0] == "-n" && w[1] == "1"));
        assert!(a.windows(2).any(|w| w[0] == "-level" && w[1] == "12"));
        assert_eq!(a.last().map(String::as_str), Some("-q"));
    }

    #[test]
    fn external_engine_config_uses_gtp() {
        let c = EdaxConfig::default_install();
        let ec = c.external_engine_config();
        assert_eq!(ec.protocol, Protocol::Gtp);
        assert_eq!(ec.board_size, BoardSize::STANDARD);
        assert_eq!(ec.command, PathBuf::from(".edax/edax"));
    }

    #[test]
    fn resolve_path_default_and_override() {
        assert_eq!(resolve_edax_path(None), PathBuf::from(".edax/edax"));
        assert_eq!(
            resolve_edax_path(Some(Path::new("/custom/edax"))),
            PathBuf::from("/custom/edax")
        );
    }

    #[test]
    fn availability_false_for_missing() {
        assert!(!EdaxConfig::new("/definitely/not/here/edax", 1).is_available());
    }
}
