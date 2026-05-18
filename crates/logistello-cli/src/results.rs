//! `results/` run-directory layout (design doc `Logistello.md` §4.2;
//! Phase 9b).
//!
//! Each invocation that produces measured artifacts writes them under a
//! timestamped directory `results/<YYYYMMDD_HHMMSS>/…` and (best-effort)
//! repoints a `results/latest` symlink at it. `results/` is gitignored
//! (reproducible via the documented commands), so this is intentionally
//! minimal: a directory + a `latest` convenience link. Callers may also
//! still write to an explicit `--output` path; this just provides the
//! canonical timestamped home and the `latest` pointer.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Creates `root/<timestamp>/` and returns it, refreshing `root/latest`
/// to point at it (best-effort: a failed symlink is not fatal — e.g. on
/// filesystems without symlink support the directory is still usable).
///
/// The timestamp is `YYYYMMDD_HHMMSS` (UTC) so directories sort
/// chronologically. If two runs land in the same second a `_<n>` suffix is
/// appended so an existing run is never clobbered.
///
/// # Errors
///
/// Returns an error only if the timestamped directory itself cannot be
/// created.
pub fn new_run_dir(root: &Path) -> Result<PathBuf> {
    use chrono::Utc;
    std::fs::create_dir_all(root).with_context(|| format!("create {}", root.display()))?;
    let stamp = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let mut dir = root.join(&stamp);
    let mut n = 1;
    while dir.exists() {
        dir = root.join(format!("{stamp}_{n}"));
        n += 1;
    }
    std::fs::create_dir_all(&dir).with_context(|| format!("create {}", dir.display()))?;
    refresh_latest(root, &dir);
    Ok(dir)
}

/// Best-effort: repoint `root/latest` at `target` (relative link so the
/// tree stays relocatable). Silently ignores failures.
fn refresh_latest(root: &Path, target: &Path) {
    let link = root.join("latest");
    let _ = std::fs::remove_file(&link);
    let name = target.file_name().unwrap_or_default();
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(name, &link);
    }
    #[cfg(not(unix))]
    {
        // No symlink: drop a text pointer instead (still discoverable).
        let _ = std::fs::write(root.join("latest.txt"), name.to_string_lossy().as_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_run_dir_is_unique_and_links_latest() {
        let base = std::env::temp_dir().join(format!("logi_results_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let a = new_run_dir(&base).unwrap();
        let b = new_run_dir(&base).unwrap();
        assert!(a.is_dir() && b.is_dir());
        assert_ne!(a, b, "two runs must not share a directory");
        #[cfg(unix)]
        {
            let link = base.join("latest");
            assert!(link.exists(), "results/latest must resolve");
            let resolved = std::fs::canonicalize(&link).unwrap();
            // latest points at the most recent run (b).
            assert_eq!(resolved, std::fs::canonicalize(&b).unwrap());
        }
        let _ = std::fs::remove_dir_all(&base);
    }
}
