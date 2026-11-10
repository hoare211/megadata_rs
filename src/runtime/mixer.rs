use crate::builder::metadata::StorageFormat;
use crate::runtime::stream::{BatchData, ReaderExit, StreamReader};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand_chacha::ChaCha12Rng;

// ★ 评审 §3.8(d)：原先此处有 `#[allow(dead_code)] pub struct MixingStrategy` +
//   `from_temperature(token_counts, temperature)`（注释："公开 API：Python 侧温度权重
//   未来可下沉到 Rust"）。已删除，理由三条：
//     1) `runtime::mixer` 是 `pub(crate)` 模块 → 该"公开 API"**外部根本无法访问**，
//        不是对外承诺，只是没有调用方的死代码；
//     2) 温度加权公式已在 Python 侧完整实现（`engine.py::_effective_weights`：
//        `base_i ** (1/T)` 再归一化），Rust 侧这份是**同一公式的第二份实现**——
//        两份实现漂移是典型隐患，正是 §3.5 文档/代码不符问题的同源风险；
//     3) 零测试覆盖（无任何断言验证权重归一化或 T→1 极限）。
//   若将来真要下沉，应届时连同测试一起搬，而不是先留一份无测试的影子实现。

/// Mixer 断点续训快照（真 O(1) 恢复，借鉴 streaming 五元组 checkpoint 思想）：
/// 恢复 = set_word_pos + 恢复各子流游标，不再重放 N 步 RNG 采样。
///
/// v3（R4/B2'）：游标是**逐子流 `(epoch, in_epoch)`**（"下一产出位置"口径），
/// 取代 v2 的"全局已消费计数"——后者在 R2（按 epoch 定位）之后无法表达，
/// 且恢复时会撞上 `in_epoch >= epoch_total(epoch)` 的硬校验。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MixerState {
    /// 已**成功产出**的总步数（只在产出后推进，采样失败不计数——C3/B3）
    pub steps_yielded: u64,
    /// 逐子流 `(epoch, in_epoch)` 游标（下一产出位置；耗尽子流归一化为 `(e+1, 0)`）
    pub sub_stream_cursors: Vec<(u64, u64)>,
    /// ChaCha12 RNG 的字位置（32-bit word 单位）：恢复后继续产生与原 run
    /// 完全一致的采样序列。
    pub rng_word_pos: u128,
}

pub struct StreamMixer {
    streams: Vec<StreamReader>,
    distribution: WeightedIndex<f64>,
    rng: ChaCha12Rng,
    global_seed: u64,
    steps_yielded: u64,
    /// 逐子流 `(epoch, in_epoch)` 游标。**Rust 维护**（seek/wrap/产出时更新），
    /// Python 只读（`state_dict`）——O(1) 恢复不跨语言拼状态。
    sub_stream_cursors: Vec<(u64, u64)>,
}

impl StreamMixer {
    /// 构造混采器。
    ///
    /// 权重必须是"长度与子流数一致、至少一个正数、不含负数/NaN"（`WeightedIndex` 的要求）。
    /// 旧实现直接 `.unwrap()`：空数据集（token 数全 0）算出的全 0 权重会让它 panic，
    /// 并冒泡成 pyo3 `PanicException`（复核问题 9）。现在返回可操作错误。
    ///
    /// `start_epoch`：所有子流的共同起始轮号（C25/H1 —— `MixBatchStream(epoch=k)` 的落点）。
    /// 子流是 Rust 私有字段、Python 无句柄，只能在构造时于 Rust 内部把全部子流
    /// seek 到 `(start_epoch, 0)`。
    pub fn new(
        streams: Vec<StreamReader>,
        weights: Vec<f64>,
        global_seed: u64,
        start_epoch: u64,
    ) -> Result<Self, String> {
        if weights.len() != streams.len() {
            return Err(format!(
                "weights count ({}) does not match sub-stream count ({}), check datasets_config",
                weights.len(),
                streams.len()
            ));
        }
        let distribution = WeightedIndex::new(&weights).map_err(|e| {
            format!(
                "invalid weights ({e}): need at least one positive weight, and no negatives/NaN. Current weights: {weights:?}.\
                 If using temperature to auto-balance by token count, check whether each dataset's global_stats.total_tokens is 0."
            )
        })?;

        let mut mixer = Self::assemble(streams, distribution, global_seed);
        if start_epoch > 0 {
            mixer.seek_all_to(start_epoch)?;
        }
        Ok(mixer)
    }

    /// 内部组装（不做参数校验）：`new` 与单测共用。
    /// 单测用"空子流 + 权重"验证 RNG / 快照等价性，需要绕过"权重数 == 子流数"校验。
    fn assemble(
        streams: Vec<StreamReader>,
        distribution: WeightedIndex<f64>,
        global_seed: u64,
    ) -> Self {
        Self {
            sub_stream_cursors: vec![(0, 0); streams.len()],
            streams,
            distribution,
            rng: ChaCha12Rng::seed_from_u64(global_seed),
            global_seed,
            steps_yielded: 0,
        }
    }

    /// 把所有子流 seek 到 `(epoch, 0)`（`MixBatchStream(epoch=k)` 的构造期路径）
    fn seek_all_to(&mut self, epoch: u64) -> Result<(), String> {
        for i in 0..self.streams.len() {
            self.seek_stream(i, epoch, 0)
                .map_err(|e| format!("Failed to seek sub-stream {i} to epoch {epoch}: {e}"))?;
        }
        Ok(())
    }

    #[allow(dead_code)] // 保留为公开诊断 API
    pub fn stream_format(&self, stream_idx: usize) -> StorageFormat {
        self.streams[stream_idx].manifest.format.clone()
    }

    /// **R4②（C3/B3）**：纯采样——一次 rng 抽取，选定子流下标，**不推进任何计数**。
    ///
    /// 计数只在子流**成功产出**后由 `on_batch_produced` 推进：采样到已耗尽子流
    /// （`SampledExhausted`）不计数，否则恢复时会固定多 1（每轮静默漏一个 batch）。
    pub fn sample_stream_idx(&mut self) -> usize {
        self.distribution.sample(&mut self.rng)
    }

    /// 子流 `stream_idx` 成功产出一个 batch 后调用：推进步数与该子流的轮内位置。
    pub fn on_batch_produced(&mut self, stream_idx: usize) {
        self.steps_yielded += 1;
        if let Some((_, in_epoch)) = self.sub_stream_cursors.get_mut(stream_idx) {
            *in_epoch += 1;
        }
    }

    /// 在指定 stream 上非阻塞地尝试取一个 batch（最多阻塞 POLL_TIMEOUT）。
    ///
    /// 语义同 `StreamReader::try_next`（R1 后 `Err(Eof)` = **该子流本轮读完**，
    /// 是正常事件而非 mixer 结束；`ReaderExit::Fatal` 必须上报）。
    pub fn try_next_at(&self, stream_idx: usize) -> Result<Option<BatchData>, ReaderExit> {
        self.streams[stream_idx].try_next()
    }

    /// **R4①**：子流 i 本轮是否已读完（透出 `at_eof`）。
    /// Python 用它零消耗地判定轮边界 / 重入时的 wrap-all 范围（B9'）。
    pub fn is_exhausted(&self, i: usize) -> bool {
        self.streams[i].at_eof()
    }

    /// 所有已耗尽的字流下标（自然模式重入 wrap-all 的便捷入口）
    pub fn exhausted_streams(&self) -> Vec<usize> {
        (0..self.streams.len())
            .filter(|&i| self.streams[i].at_eof())
            .collect()
    }

    /// **R4④ 内部原语**：把子流 i seek 到指定 `(epoch, in_epoch)` 并同步游标。
    /// 构造期 start_epoch / 自然模式 wrap-all / 确定性模式 wrap 三处共用。
    pub fn seek_stream(&mut self, i: usize, epoch: u64, in_epoch: u64) -> std::io::Result<()> {
        // ★ 评审 §3.2：`debug_assert!` 在 release 下被编译掉，越界会变成 `self.streams[i]`
        //   的索引 panic —— 而本函数可经 `RustStreamMixer.seek_stream` 从 Python 直接
        //   到达，panic 会以 pyo3 `PanicException` 越过 FFI 边界。
        //   `mixer.rs` 里`update_weights` / `load_state` / `new` 三处都做了运行时长度
        //   校验（且本文件 181-183 行的注释明确担心过这个问题），唯独这里与下面的
        //   `wrap_stream` 漏了。现在统一改为运行时检查并返回可操作的错误。
        if i >= self.streams.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "sub-stream index {i} out of range ({} sub-streams in total)",
                    self.streams.len()
                ),
            ));
        }
        self.streams[i].seek_to_epoch_batch(epoch, in_epoch)?;
        if let Some(c) = self.sub_stream_cursors.get_mut(i) {
            *c = (epoch, in_epoch);
        }
        Ok(())
    }

    /// `seek_stream` 的便捷封装：把子流 i 推进到它自己的下一轮开头（wrap 语义）。
    pub fn wrap_stream(&mut self, i: usize) -> std::io::Result<()> {
        // ★ 评审 §3.2：同`seek_stream`，release 下不能只靠 debug_assert。
        if i >= self.sub_stream_cursors.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "sub-stream index {i} out of range ({} sub-streams in total)",
                    self.sub_stream_cursors.len()
                ),
            ));
        }
        let next_epoch = self.sub_stream_cursors[i].0 + 1;
        self.seek_stream(i, next_epoch, 0)
    }

    /// 动态更新采样权重
    pub fn update_weights(&mut self, new_weights: &[f64]) -> Result<(), String> {
        // 必须校验长度：`WeightedIndex::new` 只校验权重合法性，不校验长度与子流数匹配。
        // 短权重会让 sample() 恒落在前几个下标 → 其余子流被静默永久屏蔽；
        // 长权重会让 sample() 返回越界下标 → try_next_at 直接索引 panic（穿越 PyO3 边界）。
        if new_weights.len() != self.streams.len() {
            return Err(format!(
                "weights count ({}) does not match sub-stream count ({})",
                new_weights.len(),
                self.streams.len()
            ));
        }
        self.distribution = WeightedIndex::new(new_weights)
            .map_err(|e| format!("Failed to update weights (all zeros or negative?): {}", e))?;
        Ok(())
    }

    /// 子流数量（绑定层越界校验用）
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// 各子流"当前所在轮"的真实轮长（名义 `len()` 与跨 rank 校验用）。
    pub fn substream_epoch_totals(&self) -> Vec<u64> {
        self.streams
            .iter()
            .zip(self.sub_stream_cursors.iter())
            .map(|(s, (epoch, _))| s.epoch_total_batches(*epoch))
            .collect()
    }

    /// epoch-0 的名义总 batch 数（跨子流求和；已降级为"epoch-0 名义值"）
    pub fn total_batches_for_rank(&self) -> u64 {
        self.streams
            .iter()
            .map(|s| s.total_batches_for_rank())
            .sum()
    }

    /// 导出 O(1) 恢复快照（与 Python MixBatchStream.state_dict 配合）
    ///
    /// **C22/G3**：耗尽子流的游标归一化为 `(epoch+1, 0)`（"下一产出位置"口径）——
    /// 若写 `(e, L_e)`，恢复时会被 R2 的 `in_epoch >= epoch_total(epoch)` 直接拒绝。
    pub fn state_dict(&self) -> MixerState {
        let cursors = (0..self.streams.len())
            .map(|i| {
                if self.streams[i].at_eof() {
                    (self.sub_stream_cursors[i].0 + 1, 0)
                } else {
                    self.sub_stream_cursors[i]
                }
            })
            .collect();
        MixerState {
            steps_yielded: self.steps_yielded,
            sub_stream_cursors: cursors,
            rng_word_pos: self.rng.get_word_pos(),
        }
    }

    /// 从快照 O(1) 恢复：RNG 直接跳到字位置，各子流 seek 到各自已消费数。
    /// 注意：若运行期用 update_weights 改过权重，恢复应使用快照时刻的权重
    /// （Python 侧 state_dict 已一并保存 weights）。
    pub fn load_state(
        &mut self,
        state: &MixerState,
        weights: &[f64],
        streams: Option<Vec<StreamReader>>,
    ) -> Result<(), String> {
        if state.sub_stream_cursors.len() != self.streams.len() && streams.is_none() {
            return Err(format!(
                "sub_stream_cursors length {} does not match sub-stream count {}",
                state.sub_stream_cursors.len(),
                self.streams.len()
            ));
        }
        if let Some(new_streams) = streams {
            if new_streams.len() != state.sub_stream_cursors.len() {
                return Err(format!(
                    "sub-stream count {} does not match snapshot sub_stream_cursors length {}",
                    new_streams.len(),
                    state.sub_stream_cursors.len()
                ));
            }
            self.streams = new_streams;
        }
        // 权重长度必须与（可能刚替换的）子流数一致，否则 sample() 可能返回越界下标
        // 导致 try_next_at panic 穿越 PyO3 边界（同 update_weights 的校验）。
        if weights.len() != self.streams.len() {
            return Err(format!(
                "weights count ({}) does not match sub-stream count ({})",
                weights.len(),
                self.streams.len()
            ));
        }
        self.distribution =
            WeightedIndex::new(weights).map_err(|e| format!("Failed to restore weights: {e}"))?;
        self.rng = ChaCha12Rng::seed_from_u64(self.global_seed);
        self.rng.set_word_pos(state.rng_word_pos);
        self.steps_yielded = state.steps_yielded;
        self.sub_stream_cursors = state.sub_stream_cursors.clone();

        // 逐子流按 (epoch, in_epoch) 定位（R2 的同一 API）
        let cursors = self.sub_stream_cursors.clone();
        for (i, (epoch, in_epoch)) in cursors.iter().enumerate() {
            self.streams[i]
                .seek_to_epoch_batch(*epoch, *in_epoch)
                .map_err(|e| format!("Failed to restore sub-stream {i} to (epoch={epoch}, in_epoch={in_epoch}): {e}"))?;
        }
        Ok(())
    }

    /// 是否任一子流被请求取消。
    pub fn is_cancelled(&self) -> bool {
        self.streams.iter().any(|s| s.is_cancelled())
    }

    /// 取消所有子流。
    pub fn cancel(&self) {
        for s in &self.streams {
            s.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixer_state_o1_resume_equivalence() {
        // 采样 1000 步 → 快照 → 恢复 → 后续采样序列必须与基准完全一致
        let weights = vec![0.7, 0.3];
        let mix =
            |seed| StreamMixer::assemble(Vec::new(), WeightedIndex::new(&weights).unwrap(), seed);
        // v3：采样与"成功产出"是两步（C18），计数只在产出后推进
        let step = |m: &mut StreamMixer| {
            let i = m.sample_stream_idx();
            m.on_batch_produced(i);
            i
        };

        let mut baseline = mix(1024);
        for _ in 0..1000 {
            step(&mut baseline);
        }

        let state = baseline.state_dict();
        // 真实场景 streams 非空；这里聚焦 RNG 序列等价性。
        // 注意：load_state 现在校验 weights 数 == 子流数（防越界 panic），
        // 空子流 + 非空权重的单测专用路径不满足该不变量，因此手动恢复
        // RNG/计数/游标（load_state 内部即做这三件事 + 逐子流 seek）。
        let mut baseline2 = mix(1024);
        baseline2.rng = ChaCha12Rng::seed_from_u64(1024);
        baseline2.rng.set_word_pos(state.rng_word_pos);
        baseline2.steps_yielded = state.steps_yielded;
        baseline2.sub_stream_cursors = state.sub_stream_cursors.clone();

        let tail_base: Vec<usize> = (0..50).map(|_| step(&mut baseline)).collect();
        let tail_rest: Vec<usize> = (0..50).map(|_| step(&mut baseline2)).collect();
        assert_eq!(
            tail_base, tail_rest,
            "sampling sequence inconsistent after O(1) restore"
        );
        assert_eq!(baseline2.state_dict().steps_yielded, 1050);
    }

    /// C3/B3：计数只在**成功产出**后推进——采样到已耗尽子流不该计数。
    #[test]
    fn test_mixer_counts_only_after_successful_production() {
        let weights = vec![1.0];
        let mut m = StreamMixer::assemble(Vec::new(), WeightedIndex::new(&weights).unwrap(), 7);
        let i = m.sample_stream_idx();
        assert_eq!(i, 0, "single sub-stream should always sample index 0");
        assert_eq!(
            m.state_dict().steps_yielded,
            0,
            "sampling alone must not advance steps (count timing)"
        );
        m.on_batch_produced(i);
        assert_eq!(
            m.state_dict().steps_yielded,
            1,
            "steps advance only after successful production"
        );
    }

    /// 回归（复核问题 9）：参数非法要给可操作错误，而不是让 `WeightedIndex::new().unwrap()`
    /// 一路 panic 到 pyo3 `PanicException`。
    #[test]
    fn test_mixer_new_validates_weights() {
        // 注意：StreamMixer 未实现 Debug，不能用 unwrap_err()
        let err = match StreamMixer::new(Vec::new(), vec![0.5], 1, 0) {
            Ok(_) => panic!("mismatched weights vs sub-stream count should raise an error"),
            Err(e) => e,
        };
        assert!(
            err.contains("does not match"),
            "mismatched weights vs sub-stream count should be explained clearly: {err}"
        );

        let err = match StreamMixer::new(Vec::new(), vec![], 1, 0) {
            Ok(_) => panic!("empty weights should raise an error"),
            Err(e) => e,
        };
        assert!(
            err.contains("invalid weights"),
            "empty/invalid weights should give an actionable error: {err}"
        );
    }

    #[test]
    fn test_mixer_state_size_is_tiny() {
        // 置换表/无状态设计的意义：checkpoint 从 GB 级 buffer → 几十字节
        let weights = vec![0.5, 0.5];
        let mut m = StreamMixer::assemble(vec![], WeightedIndex::new(&weights).unwrap(), 1);
        for _ in 0..100 {
            let i = m.sample_stream_idx();
            m.on_batch_produced(i);
        }
        let json = serde_json::to_vec(&m.state_dict()).unwrap();
        assert!(
            json.len() < 200,
            "MixerState serialization should be < 200B, got {}B",
            json.len()
        );
    }

    /// 回归（二轮 P1-2）：权重长度与子流数不匹配必须显式报错——
    /// 短权重静默屏蔽子流、长权重 sample() 越界 panic 穿越 PyO3 边界。
    #[test]
    fn test_update_weights_and_load_state_validate_length() {
        let weights = vec![0.5, 0.5];
        let mut m = StreamMixer::assemble(vec![], WeightedIndex::new(&weights).unwrap(), 7);
        let err = match m.update_weights(&[1.0]) {
            Ok(_) => panic!("short weights should raise an error"),
            Err(e) => e,
        };
        assert!(
            err.contains("does not match"),
            "length mismatch should be explained: {err}"
        );

        let state = m.state_dict();
        let err = match m.load_state(&state, &weights, None) {
            Ok(_) => panic!("weights longer than sub-stream count should raise an error"),
            Err(e) => e,
        };
        assert!(
            err.contains("does not match"),
            "load_state length mismatch should be explained: {err}"
        );
    }

    /// ★ 评审 §3.2 回归：下标越界必须返回可操作错误，**不能**变成索引 panic。
    ///
    /// 旧实现两处都是 `debug_assert!`（release 下被编译掉），Python 侧可直接
    /// 调`wrap_stream(i)` / `seek_stream(i, ..)` 触发 `self.streams[i]` 越界，
    /// panic 会以 pyo3 `PanicException` 越过 FFI 边界。
    #[test]
    fn out_of_range_sub_stream_index_is_an_error_not_a_panic() {
        let weights = vec![0.5, 0.5];
        let mut m = StreamMixer::assemble(vec![], WeightedIndex::new(&weights).unwrap(), 7);

        let err = match m.seek_stream(2, 0, 0) {
            Ok(()) => panic!("out-of-range seek_stream should return Err, not panic"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("out of range"),
            "seek_stream out-of-range must be explained: {err}"
        );

        let err = match m.wrap_stream(9) {
            Ok(()) => panic!("out-of-range wrap_stream should return Err, not panic"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("out of range"),
            "wrap_stream out-of-range must be explained: {err}"
        );
    }

    /// Mixer wrap 端到端（二轮测试缺口）：真实 StreamReader 子流上
    /// "消费完本轮 → wrap → 续读到下一轮数据"全链路（此前只有空 streams 的
    /// RNG 等价性单测，wrap → 续读 → 轮长变化无任何测试）。
    #[test]
    fn mixer_wrap_then_next_epoch_reads_data_end_to_end() {
        use crate::runtime::stream::StreamReaderOptions;

        const NUM_BATCHES: usize = 4;
        let (bs, msl) = (2usize, 4usize);

        // 本地 Memmap 数据集（同 stream.rs 测试的 make_local_dataset 做法）
        let dir = std::env::temp_dir().join(format!("megadata_mixer_wrap_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut manifest = crate::builder::metadata::DatasetManifest::new(
            crate::builder::metadata::StorageFormat::Memmap,
            msl,
        );
        manifest.global_stats.batch_size = bs;
        manifest.shards.push(crate::builder::metadata::ShardMeta {
            shard_id: 0,
            filename: "shard_00000.bin".to_string(),
            num_batches: NUM_BATCHES,
            size_bytes: (bs * msl * 4 * NUM_BATCHES) as u64,
            hashes: Default::default(),
            align_aud_range: None,
        });
        manifest
            .save_to_file(dir.join("index.json").to_str().unwrap())
            .unwrap();
        std::fs::write(
            dir.join("shard_00000.bin"),
            vec![0u8; bs * msl * 4 * NUM_BATCHES],
        )
        .unwrap();

        let reader = StreamReader::with_options(
            dir.join("index.json").to_str().unwrap(),
            bs,
            msl,
            1,
            0,
            42,
            StreamReaderOptions::default(),
        )
        .unwrap();
        let mut mixer = StreamMixer::new(vec![reader], vec![1.0], 42, 0)
            .expect("single-stream mixer should construct");

        // 1. 消费完 epoch 0 的全部 batch
        let mut produced = 0;
        while produced < NUM_BATCHES {
            match mixer.try_next_at(0) {
                Ok(Some(_)) => {
                    mixer.on_batch_produced(0);
                    produced += 1;
                }
                Ok(None) | Err(ReaderExit::Eof) => break,
                Err(ReaderExit::Fatal(msg)) => panic!("epoch 0 read failed: {msg}"),
            }
        }
        assert_eq!(produced, NUM_BATCHES, "epoch 0 should yield all batches");
        // 预取线程在送完最后一个 batch 后的"下一轮迭代"才会置 at_eof（并发管道的正常
        // 一拍延迟）。直接断言会与后台线程的设置时机竞争（flaky）——这里给最多 2s
        // 的轮询窗口（与 stream.rs next_within 同款模式），消除竞态而不掩盖真死锁。
        let eof_deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while !mixer.is_exhausted(0) && std::time::Instant::now() < eof_deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            mixer.is_exhausted(0),
            "sub-stream should be at EOF after consuming epoch 0 (within 2s)"
        );

        // 2. wrap → 游标推进到 epoch 1 → 续读必须拿到数据（不是 EOF/超时）
        mixer
            .wrap_stream(0)
            .expect("wrap after exhaustion should succeed");
        assert_eq!(
            mixer.sub_stream_cursors[0],
            (1, 0),
            "cursor must move to (epoch 1, 0)"
        );
        match mixer.try_next_at(0) {
            Ok(Some(_)) => mixer.on_batch_produced(0),
            Ok(None) | Err(ReaderExit::Eof) => {
                panic!("epoch 1 must yield data after wrap, got EOF")
            }
            Err(ReaderExit::Fatal(msg)) => panic!("epoch 1 read failed after wrap: {msg}"),
        }

        // 3. epoch 1 同样应能读满（轮长跨 epoch 稳定：单 shard 固定 batch 数）
        let mut produced1 = 1;
        while produced1 < NUM_BATCHES {
            match mixer.try_next_at(0) {
                Ok(Some(_)) => {
                    mixer.on_batch_produced(0);
                    produced1 += 1;
                }
                Ok(None) | Err(ReaderExit::Eof) => break,
                Err(ReaderExit::Fatal(msg)) => panic!("epoch 1 read failed: {msg}"),
            }
        }
        assert_eq!(
            produced1, NUM_BATCHES,
            "epoch 1 should yield the same batch count"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
