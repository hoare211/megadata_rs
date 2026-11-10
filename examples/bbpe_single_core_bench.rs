//! 单核 bbpe encode microbench（0.6 Tier-0 快赢项 / #37 定级依据）。
//!
//! 唯一现有数字是「合成吞吐 240k tok/s/线程」——本实验用真实 encode 直接测它，
//! 在一切 #37 改造之前定级：
//!
//! - 单核 ≈240k tok/s ⇒ 并行度已打满，#37 上限只有消除 IO/适配开销的 2–3×，
//!   3–10× 不可能，真正解法是换分词器后端（另一个 XL）→ #37 定级 **M**；
//! - 单核 ≥2 MB/s（≈500k+ tok/s）⇒ 有 2× 并行度损失可捞 → #37 定级 **L**。
//!
//! ## 已产出结论（2026-10-01，本机 80 逻辑核 / 2 NUMA）
//!
//! - 单核 ≈2.0M tok/s（≈4.8 MB/s），**是 240k 参考值的 ~8 倍** ⇒ #37 定级 **L**。
//! - rayon 并行对照：26 线程仅 3.3× 加速 / 13% 并行效率；**8 线程 7.2×** ⇒ 线程甜点
//!   在 ~8，26 线程落在性能洼地（差 ~2.2×）。
//! - **交叉验证**：26 线程纯 encode ≈6.5 Mtok/s，与 phase9 端到端 6.22 Mtok/s 几乎相等
//!   ⇒ 端到端吞吐被 encode 的并行上限锁死，IO/parse/pack 非主要约束。
//!
//! ⚠️ 测量前提：必须用 `--release`。debug 构建未优化，读数会低约 10× 并导致定级误判。
//! ⚠️ 重复次数：本工具单次执行只跑一轮（串行/并行各一次）；下文结论中的 "best-of-3"
//!    系手工重复运行 3 次取最优的历史读数，不是本工具内置能力（T4 repeats 归 Tier-1
//!    bench 补强包）。需要方差数据时请重复执行或先落 T4。
//! ⚠️ 外推限制：本机仅有 vocab 6400 的小词表分词器样本，且默认语料为合成文本；
//!    生产若用 32k–128k 词表 + 真实杂乱语料，单核绝对值会下降，但不足以翻转 L 级结论。
//!
//! # 用法
//! ```sh
//! # 串行基线 + 26 线程并行对照（默认 32MB 语料）
//! cargo run --release --example bbpe_single_core_bench -- --tokenizer <path/to/tokenizer.json>
//!
//! # 扫线程甜点（本机推荐与 8 对比）
//! cargo run --release --example bbpe_single_core_bench -- \
//!   --tokenizer tok.json --threads 8
//! ```
//!
//! 不依赖 GPU / 数据集 / 训练栈，纯 CPU 微基准。RUST 端单线程计时，
//! 对齐构建管线 `tokenizer.encode_ordinary(text)` 的逐文档调用路径。

use bbpe_rs::tokenizer::BBPETokenizer;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::time::{Duration, Instant};

/// 默认语料目标字节数（32MB：足够稳定又能在 ~1 分钟内跑完）。
const DEFAULT_TARGET_BYTES: usize = 32 * 1024 * 1024;

/// 单个"文档"块的目标大小（4KB，对齐构建管线 chunk 粒度量级）。
const DOC_BLOCK_BYTES: usize = 4 * 1024;

/// 并行对照的默认线程数：对齐方案 phase9 口径（80 核机，RAYON_NUM_THREADS=26 ≈ 1/3 核预算）。
///
/// ⚠️ 本机实测（2026-10-01，32MB 语料 best-of-3）表明该值落在**性能洼地**：
/// 8 线程 ≈7.2× 加速，26 线程仅 ≈3.3× —— encode 扩展严格非线性，超过 ~8 线程后
/// 加速比**绝对下降**。保留默认 26 是为了复现 phase9 口径，判读输出会对超出甜点的
/// 情况告警。
const DEFAULT_PARALLEL_THREADS: usize = 26;

/// 本机实测的 encode 并行甜点线程数（见上）。用于判读告警。
const SWEET_SPOT_THREADS: usize = 8;

#[derive(Debug)]
struct BenchArgs {
    tokenizer: String,
    input: Option<String>,
    bytes: usize,
    /// 并行对照组使用的 rayon 线程数（基准/拆 off 组恒为 1）。
    threads: usize,
}

/// 取一个必带值的参数；缺值时友好报错（而不是 `args.remove(0)` 对空 vec panic）。
fn take_value(args: &mut Vec<String>, flag: &str) -> Result<String, String> {
    if args.is_empty() {
        return Err(format!("{flag} requires a value\n{HELP}"));
    }
    Ok(args.remove(0))
}

fn parse_args() -> Result<BenchArgs, String> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut tokenizer: Option<String> = None;
    let mut input: Option<String> = None;
    let mut bytes: usize = DEFAULT_TARGET_BYTES;
    let mut threads: usize = DEFAULT_PARALLEL_THREADS;
    while !args.is_empty() {
        let a = args.remove(0);
        match a.as_str() {
            "--tokenizer" => tokenizer = Some(take_value(&mut args, "--tokenizer")?),
            "--input" => input = Some(take_value(&mut args, "--input")?),
            "--bytes" => {
                bytes = take_value(&mut args, "--bytes")?
                    .parse::<usize>()
                    .map_err(|e| format!("--bytes: {e}"))?
            }
            "--threads" => {
                threads = take_value(&mut args, "--threads")?
                    .parse::<usize>()
                    .map_err(|e| format!("--threads: {e}"))?;
                if threads == 0 {
                    return Err("--threads must be >= 1".to_string());
                }
            }
            "-h" | "--help" => {
                println!("{HELP}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {a}\n{HELP}")),
        }
    }
    let tokenizer = tokenizer.ok_or_else(|| format!("--tokenizer is required\n{HELP}"))?;
    Ok(BenchArgs {
        tokenizer,
        input,
        bytes,
        threads,
    })
}

const HELP: &str = "\
bbpe_single_core_bench — 单核 bbpe encode 微基准（#37 定级依据）

用法:
  cargo run --release --example bbpe_single_core_bench -- \
    --tokenizer <path> [--input <file>] [--bytes N]

参数:
  --tokenizer <path>   bbpe tokenizer.json 路径（必填）
  --input <file>        语料来源；省略时用内置合成文本（重复填充）
  --bytes N             目标语料字节数（默认 32MB）
  --threads N           并行对照组的 rayon 线程数（默认 26，对齐方案 phase9 口径）

输出:
  1) 串行基线（1 线程）
  2) 并行对照（N 线程）
  3) 加速比 / 并行效率 —— 直接量化「并行度损失」，这是 #37 定级的核心直接证据
";

/// 在指定线程数的 rayon 池内，对 docs 做逐文档 encode（复刻管线 PT bbpe 路径形态）。
///
/// 管线真实形态（`builder/pipeline.rs` 相位 E）：
/// ```ignore
/// let ids_lists: Vec<_> = raw_chunk.into_par_iter().filter_map(|json_str| {
///     let text = sonic_rs::from_str(&json_str).ok()?;   // JSON parse
///     Some(tokenizer.encode_ordinary(text))             // encode
/// }).collect();
/// ```
/// 本 bench 剔除 JSON parse 后取纯 encode + `collect()` 屏障，隔离出 encode 自身的
/// 并行效率（`pool.install` 保证闭包运行在给定池内，`num_threads(1)` 即纯串行）。
fn run_encode<T: AsRef<str> + Sync>(
    pool: &rayon::ThreadPool,
    tokenizer: &BBPETokenizer,
    docs: &[T],
) -> (u64, Duration) {
    let start = Instant::now();
    let total: u64 = pool.install(|| {
        docs.par_iter()
            .map(|d| tokenizer.encode_ordinary(d.as_ref()).len() as u64)
            .sum()
    });
    (total, start.elapsed())
}

/// 构造 ~target_bytes 的语料：从输入文件读取后按行重复；无输入则用合成文本。
fn build_corpus(input: &Option<String>, target_bytes: usize) -> (String, usize) {
    if let Some(path) = input {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read input {path}: {e}; falling back to synthetic text");
            String::new()
        });
        if src.is_empty() {
            return synthetic_corpus(target_bytes);
        }
        let mut buf = String::with_capacity(target_bytes);
        while buf.len() < target_bytes {
            buf.push_str(&src);
            if !src.ends_with('\n') {
                buf.push('\n');
            }
        }
        let cut = buf
            .char_indices()
            .nth(target_bytes)
            .map(|(i, _)| i)
            .unwrap_or(buf.len());
        buf.truncate(cut);
        let len = buf.len();
        (buf, len)
    } else {
        synthetic_corpus(target_bytes)
    }
}

/// 合成语料：混合长短句 + 重复，覆盖常见 BPE merge 路径。
fn synthetic_corpus(target_bytes: usize) -> (String, usize) {
    let sentences = [
        "The quick brown fox jumps over the lazy dog near the riverbank at dawn. ",
        "In machine learning, tokenization is the first step of preprocessing text into discrete units. ",
        " megadata_rs 构建 PT 语料时 bbpe_rs 对每个文档调用 encode_ordinary，par_iter 并行编码。 ",
        "FlashAttention varlen 依赖 cu_seqlens 描述文档段边界；padding 分散在每行末尾。 ",
        "sha256 增量计算消除了每 shard 5 文件全量回读的写线程尾阻塞。 ",
        "— mmaps 真正的危险路径只有外部 truncate；unlink 不会 SIGBUS（inode 被映射保活）。 ",
    ];
    let mut buf = String::with_capacity(target_bytes);
    let mut i = 0;
    while buf.len() < target_bytes {
        buf.push_str(sentences[i % sentences.len()]);
        i += 1;
    }
    let cut = buf
        .char_indices()
        .nth(target_bytes)
        .map(|(i, _)| i)
        .unwrap_or(buf.len());
    buf.truncate(cut);
    let len = buf.len();
    (buf, len)
}

/// 把语料切成 ~DOC_BLOCK_BYTES 的文档块。
///
/// 优先在换行处切（对齐真实 JSONL 的逐文档边界）；无换行时落到 UTF-8 字符边界
/// 切（合成语料也能产生多块，避免"一个 2MB 巨文档"把单次 encode 的内存开销
/// 混入吞吐测量）。切点永不落在 UTF-8 字符中间。
fn split_docs(corpus: &str) -> Vec<&str> {
    let bytes = corpus.as_bytes();
    if corpus.is_empty() {
        return Vec::new();
    }
    let mut docs = Vec::new();
    let mut start = 0usize;
    let mut acc = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        acc += 1;
        // 推进到下一个 UTF-8 字符边界（多字节字符的安全推进）
        let next = i + utf8_len_from_lead(bytes[i]);
        if bytes[i] == b'\n' && acc >= DOC_BLOCK_BYTES {
            docs.push(&corpus[start..=i]);
            start = next;
            acc = 0;
        }
        i = next;
    }
    if start < corpus.len() {
        docs.push(&corpus[start..]);
    }
    // 若没切出多块（无换行 + 小语料），用字符边界强制切分以保证测量贴近"逐文档"路径
    if docs.len() <= 1 && corpus.len() > DOC_BLOCK_BYTES {
        docs.clear();
        let mut s = 0usize;
        while s < corpus.len() {
            let target = (s + DOC_BLOCK_BYTES).min(corpus.len());
            // 回退到字符边界
            let mut e = target;
            while e > s && e < corpus.len() && !corpus.is_char_boundary(e) {
                e -= 1;
            }
            e = e.max(s + 1).min(corpus.len());
            docs.push(&corpus[s..e]);
            s = e;
        }
    }
    docs
}

/// 返回 UTF-8 首字节指示的字符字节长度（非首字节返回 1，调用前已保证 i 在字符首字节）。
fn utf8_len_from_lead(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else if b >> 3 == 0b11110 {
        4
    } else {
        1
    } // 续字节或畸形，按 1 推进（合成语料为合法 UTF-8，不会命中）
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    println!("[load] tokenizer: {}", args.tokenizer);
    let tokenizer = match BBPETokenizer::load(&args.tokenizer) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to load bbpe tokenizer: {e}");
            std::process::exit(1);
        }
    };
    let vocab_size = tokenizer.state.vocab.len();
    println!("[load] vocab size: {vocab_size}");

    let (corpus, corpus_bytes) = build_corpus(&args.input, args.bytes);
    println!(
        "[corpus] {corpus_bytes} bytes ({:.2} MB)",
        corpus_bytes as f64 / (1024.0 * 1024.0)
    );

    let docs = split_docs(&corpus);
    println!(
        "[corpus] split into {} documents (~{} bytes each)",
        docs.len(),
        DOC_BLOCK_BYTES
    );

    // 显式 rayon 线程池（而非全局环境变量）：单核组用 num_threads(1)，对照组用 --threads。
    // 这样两组在同一进程、同一批文档上跑，唯一变量就是线程数。
    let serial_pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("failed to build serial rayon pool");
    let parallel_pool = ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build()
        .expect("failed to build parallel rayon pool");

    // warmup：让分支预测器 / 页表热起来（~5% 语料），不计入耗时。
    let warmup_docs = docs.len() / 20 + 1;
    let warmup_tokens: u64 = docs
        .iter()
        .take(warmup_docs)
        .map(|d| tokenizer.encode_ordinary(d).len() as u64)
        .sum();
    println!("[warmup] {warmup_docs} docs / {warmup_tokens} tokens");

    // ---- 1) 串行基线（1 线程）----
    let (ser_tokens, ser_dur) = run_encode(&serial_pool, &tokenizer, &docs);
    // ---- 2) 并行对照（N 线程）----
    let (par_tokens, par_dur) = run_encode(&parallel_pool, &tokenizer, &docs);
    // 两组必须产出相同 token 数，否则是并行分块错位（兜底一致性校验）
    assert_eq!(
        ser_tokens, par_tokens,
        "serial/parallel token count mismatch"
    );

    let corpus_mb = corpus_bytes as f64 / (1024.0 * 1024.0);
    let ser_secs = ser_dur.as_secs_f64();
    let par_secs = par_dur.as_secs_f64();
    let ser_tps = if ser_secs > 0.0 {
        ser_tokens as f64 / ser_secs
    } else {
        0.0
    };
    let par_tps = if par_secs > 0.0 {
        par_tokens as f64 / par_secs
    } else {
        0.0
    };
    let ser_mbps = if ser_secs > 0.0 {
        corpus_mb / ser_secs
    } else {
        0.0
    };
    let speedup = if par_secs > 0.0 {
        ser_secs / par_secs
    } else {
        0.0
    };
    let efficiency = speedup / args.threads as f64 * 100.0;

    println!();
    println!("========== bbpe encode 微基准（串行 vs 并行）==========");
    println!("文档数:    {}", docs.len());
    println!(
        "总 token:  {ser_tokens} ({:.2} Mtok)",
        ser_tokens as f64 / 1e6
    );
    println!("语料:      {corpus_bytes} bytes ({corpus_mb:.2} MB)");
    println!();
    println!("--- 串行基线（1 线程）---");
    println!("耗时:      {ser_secs:.3} s");
    println!(
        "吞吐:      {ser_tps:.0} tokens/sec ({:.2} k tok/s/线程)",
        ser_tps / 1e3
    );
    println!("吞吐:      {ser_mbps:.2} MB/s");
    println!();
    println!("--- 并行对照（{} 线程）---", args.threads);
    println!("耗时:      {par_secs:.3} s");
    println!(
        "吞吐:      {par_tps:.0} tokens/sec ({:.2} k tok/s)",
        par_tps / 1e3
    );
    println!();
    println!("--- 并行效率（#37 直接证据）---");
    println!("加速比:    {speedup:.2}× (理想 {}×)", args.threads);
    println!("并行效率:  {efficiency:.1}%");
    println!(
        "损失:      {:.2}× 并行度未兑现",
        args.threads as f64 / speedup.max(f64::EPSILON)
    );
    println!();

    // 定级判读（对应方案 §3.2 / §一·专家1·独有贡献3）
    println!("--- #37 定级判读 ---");
    let kbps = ser_tps / 1e3;
    if (235.0..=245.0).contains(&kbps) {
        println!("单核 ≈240k tok/s ⇒ 并行度已打满。#37 上限 = 消除 IO/适配开销的 2–3×；");
        println!("                   3–10× 不可能，真正解法是换分词器后端（另一 XL）→ #37 = M。");
    } else if kbps >= 500.0 {
        println!("单核 ≥{kbps:.0}k tok/s (≥2MB/s) ⇒ 有并行度损失可捞 → #37 = L。");
    } else if kbps < 235.0 {
        println!("单核 ≈{kbps:.0}k tok/s (<240k) ⇒ encode 比预估慢，#37 天花板更低，");
        println!("                   归因表须把 encode 嫌疑权重上调。");
    } else {
        println!("单核 ≈{kbps:.0}k tok/s（240k~500k 之间）⇒ 灰色区，以 P0 归因表为准。",);
    }
    // 并行效率的独立判读：单核值不限 relevance，效率本身就是 #37 损失的直接度量
    if efficiency < 50.0 {
        println!("并行效率 {efficiency:.1}% <50% ⇒ rayon 层存在显著并行度损失，#37 值得投入；");
        println!("                   优先排查 per-doc 任务粒度 / collect() 屏障 / 虚假共享。");
    } else {
        println!("并行效率 {efficiency:.1}% ≥50% ⇒ encode 侧并行良好，瓶颈更可能在 IO/相位衔接，");
        println!("                   #37 收益有限，转向 pack/write 归因。");
    }
    // ★ 线程甜点告警：本机实测（2026-10-01）显示 encode 的扩展严格非线性——
    //   峰值落在 ~8 线程，超过后加速比**绝对下降**（allocator/内存带宽/跨 NUMA 争抢）。
    //   phase9 口径用的是 26 线程（1/3 核预算），恰好落在性能洼地里。
    if args.threads > SWEET_SPOT_THREADS {
        println!();
        println!("!!! 线程甜点告警 !!!");
        println!(
            "本线程数 ({}) 超过本机实测甜点 (~{})，加速比可能显著低于更小线程数。",
            args.threads, SWEET_SPOT_THREADS
        );
        println!("实测参照：8 线程 ≈7.2× vs 26 线程 ≈3.3×（同语料 best-of-3），相差约 2.2×。");
        println!(
            "建议：跑一次 `--threads {}` 对比，并用真实构建管线复验（见 #37 归因）。",
            SWEET_SPOT_THREADS
        );
    }
}
