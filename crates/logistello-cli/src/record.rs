//! runvault への記録の共通部分．
//!
//! 出力の置き場と同一性は runvault が持つ．タイムスタンプ付きディレクトリも
//! `results/latest` シンボリックリンクもこちらでは作らず，`Run::start` が決めた
//! run ディレクトリへ書く．
//!
//! ここに集めてあるのは 3 つ:
//!
//! - **論文メタデータ** — 本再現実装は Buro の 7 本 + 対局記録 1 本を対象に
//!   するので，`Replication` は «その run がどの論文の主張を測っているか» で
//!   選ぶ．`Replication` が持てる `Work` は 1 本なので，サブコマンド (と sweep
//!   では軸) ごとに切り替える．
//! - **条件になるファイルの同一性** — 学習済み重み・GLEM モデル・定石・ProbCut
//!   係数・Edax の実体は，**パスではなく中身が結果を決める**．旧 `config.json`
//!   はパスしか書いていなかった．`Dataset` に内容ハッシュごと載せる
//!   (`config_hash` は `data` を含むので，中身が違えば別条件になる)．
//! - **イベントの形** — 指標にできない行 (系列で識別される行・ラベル) の置き場．

use std::path::Path;

use anyhow::{Context, Result};
use runvault::{Dataset, Replication, Run, RunOptions, Target, Work};
use serde::Serialize;

use crate::elo::LevelResult;
use crate::eval_corr::CorrSample;
use crate::match_replay::ReplayRow;

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
pub const EXPERIMENT: &str = "logistello";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "logistello";

/// 乱数を引く run の分野．`master_seed` が必須になる．
///
/// `play` / `bench-search` / `eval-correlation-edax` / `sweep` は，対局相手・
/// 開始局面・標本局面のいずれかを種つきの ChaCha20 で決めるので，シードが
/// 結果を決める．
pub const DOMAIN_SIMULATION: &str = "simulation";
/// 乱数を引かない run の分野．
///
/// `match-replay` は記録された棋譜を決定的に再生するだけで，`elo-vs-edax` も
/// 自エンジンと (book 無効・単スレッドの) Edax がどちらも決定的なので，1 つも
/// 乱数を引かない．`simulation` を名乗ると，何も駆動しない `master_seed` を
/// 書くことになる．
pub const DOMAIN_ANALYSIS: &str = "analysis";

/// 時間軸の単位．オセロの «手» は runvault の語彙では `turn`．
pub const T_UNIT: &str = "turn";

/// 1 局で石を置ける回数の上限 = 初期盤面の空きマス数．
///
/// `terminal` の `budget` に使う．対局は必ず盤が埋まるか双方打てなくなるかで
/// 終わるので，`censored` は常に `false`．
pub const MAX_PLACEMENTS: u64 = 60;

/// `bench-search --speedup` の 1 条件 (full / single / mpc)．
pub const BENCH_EVENT: &str = "x.logistello.bench";
/// `elo-vs-edax` の Edax レベル 1 つぶんの集計 (旧 `elo_vs_edax.csv` の 1 行)．
pub const EDAX_LEVEL_EVENT: &str = "x.logistello.edax_level";
/// `eval-correlation-edax` の標本 1 局面 (旧 `eval_correlation_edax.csv` の 1 行)．
pub const EVAL_SAMPLE_EVENT: &str = "x.logistello.eval_sample";
/// `sweep` の子 run が持つ試行 1 本 (旧 `metrics.csv` の 1 行)．
pub const TRIAL_EVENT: &str = "x.logistello.trial";

/// 設計書 (Obsidian)．
const OBSIDIAN_NOTE: &str = "研究/98_論文レポート/80-再現実験/実装完了/logistello/設計書.md";

// ---------------------------------------------------------------------------
// 論文
// ---------------------------------------------------------------------------

/// DOI が分かっている論文は DOI を work_id にし，vault 側の同定用に paper-id も
/// 残す．DOI が無いもの (NEC の技術報告・博士論文・ICCA の一部) は paper-id を
/// work_id にする — 無い DOI を作らない．
fn work(paper_id: &str, doi: Option<&str>, title: &str, year: i64) -> Work {
    let mut w = match doi {
        Some(doi) => Work::doi(doi),
        None => Work::paper_id(paper_id),
    };
    w.paper_id = Some(paper_id.to_string());
    w.title(title).year(year).source_version("published")
}

/// Buro (1994) 博士論文．探索・評価関数を統合した実装そのものの記述．
fn phd_1994() -> Work {
    work(
        "P00001690",
        None,
        "Techniken für die Bewertung von Spielsituationen anhand von Beispielen",
        1994,
    )
}

/// Buro (1995) ProbCut．
fn probcut_1995() -> Work {
    work(
        "P00001682",
        Some("10.3233/ICG-1995-18202"),
        "ProbCut: An Effective Selective Extension of the αβ Algorithm",
        1995,
    )
}

/// Buro (1997) Multi-ProbCut + 高品質評価関数．
fn multi_probcut_1997() -> Work {
    work(
        "P00001685",
        None,
        "Experiments with Multi-ProbCut and a New High-Quality Evaluation Function for Othello",
        1997,
    )
}

/// Buro (1997) 統計に基づく評価関数．
fn eval_function_1997() -> Work {
    work(
        "P00001684",
        None,
        "An Evaluation Function for Othello Based on Statistics",
        1997,
    )
}

/// Buro (1998) GLEM．
fn glem_1998() -> Work {
    work(
        "P00001686",
        Some("10.1007/3-540-48957-6_8"),
        "From Simple Features to Sophisticated Evaluation Functions",
        1998,
    )
}

/// Buro (1999) 定石学習．
fn opening_book_1999() -> Work {
    work("P00001687", None, "Toward Opening Book Learning", 1999)
}

/// Buro (1997) Murakami 戦の対局記録．
fn murakami_match_1997() -> Work {
    work(
        "P00001689",
        Some("10.3233/ICG-1997-20311"),
        "The Othello Match of the Year: Takeshi Murakami vs. Logistello",
        1997,
    )
}

fn replication(work: Work, target: Target) -> Replication {
    Replication::new(work)
        .target(target)
        .obsidian_note(OBSIDIAN_NOTE)
}

/// `play`: 統合実装が 1 局を終局まで指せること．
pub fn replication_play() -> Replication {
    replication(
        phd_1994(),
        Target::claim(
            "integrated-engine-plays-a-full-game",
            "The learned evaluation, selective midgame search and exact endgame play one game through to terminal",
        ),
    )
}

/// `bench-search`: ProbCut が探索木を縮めること．
pub fn replication_bench_search() -> Replication {
    replication(
        probcut_1995(),
        Target::claim(
            "probcut-cuts-the-search-tree",
            "Cutting on a shallow search's value visits a fraction of the nodes the full search visits at the same nominal depth",
        ),
    )
}

/// `match-replay`: 1997 年 Murakami 戦の 6 局．
pub fn replication_match_replay() -> Replication {
    replication(
        murakami_match_1997(),
        Target::section(
            "murakami-1997-six-games",
            "The six recorded games of the 1997 Murakami match",
        ),
    )
}

/// `elo-vs-edax`: 相手エンジンに対する相対棋力．
pub fn replication_elo_vs_edax() -> Replication {
    replication(
        phd_1994(),
        Target::claim(
            "relative-strength-against-a-reference-engine",
            "The engine's playing strength is measured as a score rate against a fixed-strength reference opponent",
        ),
    )
}

/// `eval-correlation-edax`: 評価関数の値が強いエンジンの評価と相関すること．
pub fn replication_eval_correlation() -> Replication {
    replication(
        eval_function_1997(),
        Target::claim(
            "learned-evaluation-tracks-a-strong-reference",
            "A statistically learned evaluation function orders positions the way a strong reference engine does",
        ),
    )
}

/// `sweep`: 掃引しているパラメータを提案した論文を対象にする．
///
/// §6 の感度分析表は 1 行 1 パラメータで，パラメータごとに出どころの論文が違う．
/// 親と子は同じ軸なので同じものを名乗る．
pub fn replication_sweep(axis_col: &str) -> Replication {
    let (work, target) = match axis_col {
        "probcut_t" | "probcut_depth_pair" => (
            probcut_1995(),
            Target::claim(
                "probcut-parameter-sensitivity",
                "The confidence threshold T and the depth pair (d, h) decide how much of the tree ProbCut removes",
            ),
        ),
        "multi_stages" => (
            multi_probcut_1997(),
            Target::claim(
                "multi-probcut-cascade-depth",
                "A cascade of several (d, h) pairs prunes more than a single pair",
            ),
        ),
        "eval_stages" => (
            eval_function_1997(),
            Target::claim(
                "stage-dependent-evaluation",
                "Splitting the game into disc-count stages lowers the evaluation's residual error",
            ),
        ),
        "glem_max_order" | "glem_support" => (
            glem_1998(),
            Target::claim(
                "glem-feature-generation",
                "The conjunction order and the support threshold decide how many features GLEM generates",
            ),
        ),
        "book_depth" | "drawishness" => (
            opening_book_1999(),
            Target::claim(
                "opening-book-learning-parameters",
                "Self-play depth and the drawishness blend decide the learned book",
            ),
        ),
        // max_depth / tt_size / endgame_empties — 探索そのものの基盤側．
        _ => (
            phd_1994(),
            Target::claim(
                "search-parameter-sensitivity",
                "Iterative-deepening depth, transposition-table size and the exact-endgame switch decide the search cost",
            ),
        ),
    };
    replication(work, target)
}

// ---------------------------------------------------------------------------
// run の起こし方
// ---------------------------------------------------------------------------

/// どの run にも共通する部分．
///
/// `cli_args` を渡すのは，入力ファイルのパスを `parameters` にも `Dataset` の
/// `uri` にも入れないため — パスは «置き場» であって条件ではない (同じ重みを
/// 別の場所に置いただけで `config_hash` が変わってしまう) が，どう起動したかは
/// 記録に残しておきたい．
pub fn options(
    subcommand: &str,
    domain: &str,
    results_root: &Path,
    replication: Replication,
) -> RunOptions {
    RunOptions::new(EXPERIMENT, subcommand)
        .repo_id(REPO_ID)
        .domain(domain)
        .results_root(results_root)
        .cli_args(std::env::args().skip(1))
        .replication(replication)
}

// ---------------------------------------------------------------------------
// 条件になるファイル
// ---------------------------------------------------------------------------

/// 本実装が学習して作ったファイル (`LGW1` 重み・`GLM1` モデル・`OPB1` 定石・
/// ProbCut / MPC 係数)．
///
/// 語彙の `role` には «学習済みモデル» が無いので `other` にする．同定は
/// 中身のハッシュで足りる — パス (`uri`) は入れない．
pub fn learned_artifact(name: &str, path: &Path) -> Result<Dataset> {
    Dataset::new("other", name)
        .hash_of_file(path)
        .with_context(|| format!("{name} の要約に失敗: {}", path.display()))
}

/// 基準オラクルとしての Edax．
///
/// バイナリと評価重み (`eval.dat`) の **両方**が結果を決める．どちらかが違えば
/// 同じコマンドでも別の数が出るので，2 つとも内容ハッシュで載せる．
pub fn edax_datasets(binary: &Path, eval_file: &Path) -> Result<Vec<Dataset>> {
    Ok(vec![
        Dataset::new("reference", "edax-binary")
            .hash_of_file(binary)
            .with_context(|| format!("Edax バイナリの要約に失敗: {}", binary.display()))?,
        Dataset::new("reference", "edax-eval-weights")
            .hash_of_file(eval_file)
            .with_context(|| format!("Edax 評価重みの要約に失敗: {}", eval_file.display()))?,
    ])
}

/// Murakami 1997 のゴールド棋譜．
pub fn murakami_dataset(path: &Path, n_games: usize) -> Result<Dataset> {
    Ok(Dataset::eval("murakami-1997")
        .hash_of_file(path)
        .with_context(|| format!("ゴールド棋譜の要約に失敗: {}", path.display()))?
        .n(n_games as u64))
}

// ---------------------------------------------------------------------------
// イベント
// ---------------------------------------------------------------------------

/// `events.jsonl` に書く観測行 (予約キーだけ)．
///
/// `terminal` を書くなら `observation` も要る (`verify --deep` が terminal の
/// `unit_id` が observation にも現れることを要求する)．数はここには書かない．
#[derive(Serialize)]
struct ObservationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// 1 局の終端．
///
/// `outcome` は black / white / draw のラベル．勝敗は数ではないので指標にはせず，
/// ここにしか置かない．石差は run スコープの `black_score` / `white_score` が
/// 正本なので重複させない．`moves` は棋譜そのもの (ラベルの列)．
#[derive(Serialize)]
struct GameTerminalEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'a str,
    censored: bool,
    budget: u64,
    moves: &'a str,
}

/// `play` の 1 局を記録する．
///
/// `t` は «石を置いた回数»．`plies` 指標 (パスを含む記録手数) とは別の数で，
/// `budget` = 60 と比べられるのはこちらである．
pub fn log_game(
    run: &mut Run,
    unit_id: &str,
    placements: u64,
    winner: &str,
    moves: &str,
) -> Result<()> {
    run.log_event(
        "observation",
        &ObservationEvent {
            unit_id,
            t: placements,
            t_unit: T_UNIT,
        },
    )?;
    run.log_event(
        "terminal",
        &GameTerminalEvent {
            unit_id,
            t: placements,
            t_unit: T_UNIT,
            outcome: winner,
            censored: false,
            budget: MAX_PLACEMENTS,
            moves,
        },
    )?;
    Ok(())
}

/// `match-replay` の 1 判断 (旧 `murakami_replay.csv` の 1 行)．
///
/// 行は (棋譜, 手数) の 2 つで識別されるので指標にはできない —
/// `metrics.csv` の主キーは (`run_uid`, `step`, `step_unit`, `scope`, `name`) で，
/// 6 局が同じ手数で同じキーを名乗ってしまう．棋譜を `unit_id`，手数を時間軸に
/// 取った観測のパネルとして書く．
#[derive(Serialize)]
struct DecisionEvent<'a> {
    unit_id: String,
    t: u64,
    t_unit: &'static str,
    game: usize,
    logistello_move: &'a str,
    our_move: &'a str,
    matched: bool,
    our_score: i32,
    is_main: bool,
}

/// 判断のパネルをすべて書く．
pub fn log_decisions(run: &mut Run, rows: &[ReplayRow]) -> Result<()> {
    for r in rows {
        run.log_event(
            "observation",
            &DecisionEvent {
                unit_id: format!("game-{}", r.game),
                t: u64::from(r.ply),
                t_unit: T_UNIT,
                game: r.game,
                logistello_move: &r.logistello_move,
                our_move: &r.our_move,
                matched: r.matched,
                our_score: r.our_score,
                is_main: r.is_main,
            },
        )
        .with_context(|| format!("game {} ply {} の observation の記録に失敗", r.game, r.ply))?;
    }
    Ok(())
}

/// Edax の 1 レベルぶんの集計．
///
/// 時間軸を持たない複数行なので，`unit_id` を持つ実験固有イベントにする．
#[derive(Serialize)]
struct EdaxLevelEvent {
    unit_id: String,
    level: u32,
    games: u32,
    wins: u32,
    draws: u32,
    losses: u32,
    score_rate: f64,
    win_rate: f64,
    elo_delta: f64,
}

/// レベルごとの結果をすべて書く．
pub fn log_edax_levels(run: &mut Run, rows: &[LevelResult]) -> Result<()> {
    for r in rows {
        run.log_event(
            EDAX_LEVEL_EVENT,
            &EdaxLevelEvent {
                unit_id: format!("level-{}", r.level),
                level: r.level,
                games: r.games,
                wins: r.wins,
                draws: r.draws,
                losses: r.losses,
                score_rate: r.score_rate(),
                win_rate: r.win_rate(),
                elo_delta: r.elo_delta(),
            },
        )
        .with_context(|| format!("level {} の記録に失敗", r.level))?;
    }
    Ok(())
}

/// 評価値の標本 1 局面．
#[derive(Serialize)]
struct EvalSampleEvent {
    unit_id: String,
    our_score: i32,
    edax_score: i32,
}

/// 標本をすべて書く．
pub fn log_eval_samples(run: &mut Run, samples: &[CorrSample]) -> Result<()> {
    for s in samples {
        run.log_event(
            EVAL_SAMPLE_EVENT,
            &EvalSampleEvent {
                unit_id: format!("position-{}", s.idx),
                our_score: s.our_score,
                edax_score: s.edax_score,
            },
        )
        .with_context(|| format!("標本 {} の記録に失敗", s.idx))?;
    }
    Ok(())
}

/// `bench-search --speedup` の 1 条件．
///
/// 1 回の実行の中で同じ根・同じ公称深さを 3 通りの枝刈りで探索する比較なので，
/// 条件を子 run に割ると «起きていない実行» を主張することになる．条件は
/// «1 回の実行の中で観測された対象» として `unit_id` を持たせる．
#[derive(Serialize)]
struct BenchEvent {
    unit_id: &'static str,
    nodes: u64,
    search_value: i32,
}

/// 条件 1 つぶんを書く．
pub fn log_bench_condition(
    run: &mut Run,
    unit_id: &'static str,
    nodes: u64,
    search_value: i32,
) -> Result<()> {
    run.log_event(
        BENCH_EVENT,
        &BenchEvent {
            unit_id,
            nodes,
            search_value,
        },
    )
    .with_context(|| format!("{unit_id} の記録に失敗"))
}
