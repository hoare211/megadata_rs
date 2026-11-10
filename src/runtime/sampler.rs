use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
// 命名说明：本文件的 StdRng 在 rand 0.9 下就是 ChaCha12Rng 的类型别名——
// 与 mixer.rs 显式使用的 rand_chacha::ChaCha12Rng 同源同算法，两处采样
// 确定性口径一致。这里不直接写 ChaCha12Rng 是因为分区/置换不需要
// word_pos 级别的 O(1) 恢复 API（那是 mixer 断点续训专用的）。
use crate::builder::metadata::{DatasetManifest, ShardMeta};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// 分区算法（借鉴 streaming 的 partition 纯函数思想）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PartitionAlgo {
    /// 旧版：shard 按 epoch 全局打散后轮询分配。不支持弹性扩缩容
    /// （world_size 变化后每个 rank 的 shard 集合几乎全变），默认保留以兼容旧 checkpoint。
    #[default]
    RoundRobin,
    /// relaxed（streaming get_partitions_relaxed 语义）：全局 epoch 序只依赖
    /// (seed, epoch, num_canonical_nodes)，与当前拓扑无关；每个 worker 取连续切片。
    /// 换节点数/扩缩容后全局样本序不变 → 可跨拓扑断点续训。
    Relaxed,
}

impl PartitionAlgo {
    pub fn parse(s: &str) -> std::io::Result<Self> {
        match s {
            "round_robin" => Ok(PartitionAlgo::RoundRobin),
            "relaxed" => Ok(PartitionAlgo::Relaxed),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Unknown partition algorithm {other:?}, valid options: \"round_robin\" / \"relaxed\""
                ),
            )),
        }
    }
}

/// shard 访问顺序（F6）：epoch 内全局 shard 序是否按 (seed, epoch) 打散。
///
/// 与 `shuffle`（batch 级置换）**正交**：本选项只决定"哪些 shard 以什么顺序被
/// 访问"，`shuffle` 决定"每个 shard 内部的 batch 顺序"。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShardOrder {
    /// 默认：按 (global_seed, epoch) 打散（历史行为；保持默认以兼容旧 checkpoint）
    #[default]
    Random,
    /// 顺序：不打散。RoundRobin 下全局序恒为 `0..n`；Relaxed 下恒为 canonical 块内
    /// 按序（跨 epoch 稳定）。满足"逐 token 核对语料、与离线工具对齐顺序"的用法。
    Sequential,
}

impl ShardOrder {
    pub fn parse(s: &str) -> std::io::Result<Self> {
        match s {
            "random" => Ok(ShardOrder::Random),
            "sequential" => Ok(ShardOrder::Sequential),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "Unknown shard order {other:?}, valid options: \"random\" / \"sequential\""
                ),
            )),
        }
    }
}

/// 三维拓扑 World（借鉴 streaming 的 node × rank × worker）。
///
/// 解决两个问题：
/// 1. **worker 维度**：旧版 sampler 只有 (world_size, rank)，用户套
///    DataLoader(num_workers=4) 时 4 个 worker 读到完全相同的数据；
///    现在每个 (rank, worker) 二元组拿到互斥的 shard 子集。
/// 2. **环境自动探测的落点**：Python 侧从 torch.distributed env +
///    get_worker_info() 推导后经此结构传入，rank/world_size 不再容易与实际状态不一致。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct World {
    pub num_nodes: usize,
    pub ranks_per_node: usize,
    pub workers_per_rank: usize,
    /// 规范（初始）节点数：relaxed 分区的弹性锚点（streaming 的
    /// num_canonical_nodes）。None = 退化为当前 num_nodes。
    #[serde(default)]
    pub num_canonical_nodes: Option<usize>,
}

impl World {
    /// world_size = num_nodes * ranks_per_node；num_workers 即 workers_per_rank。
    pub fn from_sizes(world_size: usize, num_nodes: usize, num_workers: usize) -> Self {
        let num_nodes = num_nodes.max(1);
        // world_size 应能被 num_nodes 整除；不整除时静默取整会改变 world_size
        // （如 5/2 → ranks_per_node=2 → world_size()=4），下游报
        // "rank must be smaller than world_size" 时令人困惑——这里显式提示。
        // 当前 Python 路径恒 num_nodes=1（必整除），属防御性校验。
        if !world_size.is_multiple_of(num_nodes) {
            let ranks_per_node = (world_size / num_nodes).max(1);
            tracing::warn!(
                "world_size ({world_size}) is not divisible by num_nodes ({num_nodes}); \
                 ranks_per_node truncated to {ranks_per_node} (effective world_size = {})",
                num_nodes * ranks_per_node
            );
        }
        let ranks_per_node = (world_size / num_nodes).max(1);
        Self {
            num_nodes,
            ranks_per_node,
            workers_per_rank: num_workers.max(1),
            num_canonical_nodes: None,
        }
    }

    pub fn world_size(&self) -> usize {
        self.num_nodes * self.ranks_per_node
    }

    pub fn total_workers(&self) -> usize {
        self.world_size() * self.workers_per_rank
    }

    /// relaxed 分区用的规范节点数
    pub fn canonical_nodes(&self) -> usize {
        self.num_canonical_nodes.unwrap_or(self.num_nodes).max(1)
    }
}

#[derive(Clone)]
pub struct DistributedSampler {
    pub world: World,
    /// 全局 rank（0..world_size）；node = rank / ranks_per_node
    pub rank: usize,
    /// 本 rank 内的 DataLoader worker 编号（0..workers_per_rank）
    pub worker_id: usize,
    pub global_seed: u64,
    pub algo: PartitionAlgo,
    /// epoch 内 shard 访问顺序（F6；默认 Random = 历史行为）
    pub shard_order: ShardOrder,
}

impl DistributedSampler {
    /// 旧版兼容入口：单节点、单 worker、round-robin
    #[allow(dead_code)]
    pub fn new(world_size: usize, rank: usize, global_seed: u64) -> Self {
        assert!(
            rank < world_size.max(1),
            "rank must be smaller than world_size"
        );
        Self {
            world: World::from_sizes(world_size, 1, 1),
            rank,
            worker_id: 0,
            global_seed,
            algo: PartitionAlgo::RoundRobin,
            shard_order: ShardOrder::default(),
        }
    }

    pub fn with_world(
        world: World,
        rank: usize,
        worker_id: usize,
        global_seed: u64,
        algo: PartitionAlgo,
    ) -> Self {
        assert!(
            rank < world.world_size(),
            "rank must be smaller than world_size"
        );
        assert!(
            worker_id < world.workers_per_rank,
            "worker_id must be smaller than workers_per_rank"
        );
        Self {
            world,
            rank,
            worker_id,
            global_seed,
            algo,
            shard_order: ShardOrder::default(),
        }
    }

    /// 链式设置 epoch 内 shard 访问顺序（F6）。
    pub fn with_shard_order(mut self, order: ShardOrder) -> Self {
        self.shard_order = order;
        self
    }

    /// 本 worker 在全局三维拓扑中的扁平编号：rank * workers_per_rank + worker_id
    pub fn global_worker_index(&self) -> usize {
        self.rank * self.world.workers_per_rank + self.worker_id
    }

    /// 获取当前 Epoch 下，当前 (rank, worker) 应该处理的 Shard 列表
    pub fn get_my_shards(&self, manifest: &DatasetManifest, epoch: u64) -> Vec<ShardMeta> {
        let indices = self.my_shard_indices(manifest.shards.len(), epoch);
        indices
            .into_iter()
            .map(|i| manifest.shards[i].clone())
            .collect()
    }

    /// `get_my_shards` 的轻量版：只求本 worker 在该 epoch 的 batch 总数。
    ///
    /// 不 clone ShardMeta（含 `hashes: HashMap`）——`epoch_total_batches` 每次
    /// seek 的前台校验都会调用，全量 clone 是纯浪费的分配。
    pub fn my_shard_batch_total(&self, manifest: &DatasetManifest, epoch: u64) -> u64 {
        self.my_shard_indices(manifest.shards.len(), epoch)
            .iter()
            .map(|&i| manifest.shards[i].num_batches as u64)
            .sum()
    }

    /// 纯函数：给定 (num_shards, epoch)，返回本 worker 的全局 shard 下标列表。
    /// 所有 rank/worker 各自调用，结果互斥且并集为全集（shards >= workers 时）。
    pub fn my_shard_indices(&self, num_shards: usize, epoch: u64) -> Vec<usize> {
        let w = self.global_worker_index();
        let total_w = self.world.total_workers().max(1);
        match self.algo {
            PartitionAlgo::RoundRobin => {
                // 1. epoch 内全局打散（所有 worker 视角一致）；F6 shard_order=
                //    "sequential" 时跳过打散，全局序恒为 0..n（跨 epoch 稳定）
                let mut order: Vec<usize> = (0..num_shards).collect();
                if self.shard_order == ShardOrder::Random {
                    let mut rng = StdRng::seed_from_u64(self.global_seed.wrapping_add(epoch));
                    order.shuffle(&mut rng);
                }
                // 2. 轮询分配
                order
                    .into_iter()
                    .enumerate()
                    .filter(|(idx, _)| idx % total_w == w)
                    .map(|(_, shard)| shard)
                    .collect()
            }
            PartitionAlgo::Relaxed => {
                // F6 shard_order="sequential"：跳过 canonical 块内的 epoch 打散
                //（全局序只由 (seed, ncn) 决定，跨 epoch 稳定）；块间边界与
                // shard → canonical node 归属不变（下载局部性语义保留）。
                let order = if self.shard_order == ShardOrder::Sequential {
                    relaxed_sequential_order(
                        num_shards,
                        self.global_seed,
                        self.world.canonical_nodes(),
                    )
                } else {
                    relaxed_epoch_order(
                        num_shards,
                        self.global_seed,
                        epoch,
                        self.world.canonical_nodes(),
                    )
                };
                let (start, len) = contiguous_slice_bounds(num_shards, w, total_w);
                order.into_iter().skip(start).take(len).collect()
            }
        }
    }
}

/// relaxed 分区：全局 epoch 序（与拓扑无关）。
///
/// 双 RNG（借鉴 streaming 的 shuffle 设计）：
/// - `run_rng`：整 run 固定，决定 shard → canonical node 的归属（**跨 epoch 稳定**；
/// - `epoch_rng`：每 epoch 不同，做 canonical node 块内打散（块间边界不动，
///   位移被限制在 canonical node 边界内——py1e 的连续旋钮思想的块状简化）。
///
/// ⚠️ "下载局部性"的准确范围（复核问题 5 修正）：
///
/// 上面稳定的是 **shard → canonical node 的归属**，不是"每个 worker 的 shard 集合"。
/// worker/rank 取的是全局序里的**连续切片**，而全局序每 epoch 会重排，因此：
///
/// - canonical node 的**块内容**（哪些 shard 属于我）跨 epoch 稳定；
/// - 但块内被打散 + 再按连续切片分给各 worker ⇒ **worker 级 shard 集合每 epoch 会变**
///   （实测 ncn=1/8 时 8/8 个 worker 的集合都变；ncn=32 时 7/8 变）。
///
/// 因此"换 epoch 仍命中同一节点缓存"只对 **canonical node 粒度**成立；要落到物理机粒度，
/// 还需要把 `World.num_nodes` 按 canonical 块对齐（当前 Python 路径恒为 num_nodes=1，
/// 即整机都是本地读，局部性收益尚未兑现）。
pub fn relaxed_epoch_order(num_shards: usize, seed: u64, epoch: u64, ncn: usize) -> Vec<usize> {
    let ncn = ncn.max(1);
    let mut run_rng = StdRng::seed_from_u64(seed);
    let canon: Vec<usize> = (0..num_shards)
        .map(|_| run_rng.random_range(0..ncn))
        .collect();

    let mut epoch_rng = StdRng::seed_from_u64(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut order = Vec::with_capacity(num_shards);
    for node in 0..ncn {
        let mut block: Vec<usize> = (0..num_shards).filter(|&i| canon[i] == node).collect();
        block.shuffle(&mut epoch_rng);
        order.extend(block);
    }
    order
}

/// relaxed 分区的顺序变体（F6 shard_order="sequential"）：shard → canonical node
/// 的归属与 `relaxed_epoch_order` 使用同一 run_rng（完全一致），但块内**不打散**——
/// 全局序只由 (seed, ncn) 决定，跨 epoch 稳定（收敛到"离线工具可复现的顺序"）。
pub fn relaxed_sequential_order(num_shards: usize, seed: u64, ncn: usize) -> Vec<usize> {
    let ncn = ncn.max(1);
    let mut run_rng = StdRng::seed_from_u64(seed);
    let canon: Vec<usize> = (0..num_shards)
        .map(|_| run_rng.random_range(0..ncn))
        .collect();
    let mut order = Vec::with_capacity(num_shards);
    for node in 0..ncn {
        order.extend((0..num_shards).filter(|&i| canon[i] == node));
    }
    order
}

/// 把 n 个 shard 按连续切片分给 W 个 worker：前 n%W 个 worker 多拿一个。
/// 返回 (start, len)。
pub fn contiguous_slice_bounds(n: usize, worker: usize, total_workers: usize) -> (usize, usize) {
    let total_workers = total_workers.max(1);
    let base = n / total_workers;
    let rem = n % total_workers;
    let start = worker * base + worker.min(rem);
    let len = base + if worker < rem { 1 } else { 0 };
    (start, len)
}

/// 仿真器数据源：产出每个全局 worker "每 step 应读的全局 shard id" 序列。
///
/// 与 StreamReader 的真实读取序完全一致（同一套分区/置换纯函数），但只读
/// manifest 元数据、不做任何 IO——供 simulation 在不跑训练的前提下做
/// 带宽/缓存/饥饿的容量规划。
///
/// 返回 `out[worker][step] = 全局 shard 下标`；worker 扁平编号
/// w = rank * workers_per_rank + worker_id。
pub fn plan_shard_sequence(
    manifest: &DatasetManifest,
    world: &World,
    algo: PartitionAlgo,
    global_seed: u64,
    epochs: u64,
    shuffle: bool,
    shard_order: ShardOrder,
) -> Vec<Vec<u32>> {
    let workers_per_rank = world.workers_per_rank.max(1);
    let total_workers = world.total_workers().max(1);
    let num_shards = manifest.shards.len();
    let mut out = Vec::with_capacity(total_workers);
    for w in 0..total_workers {
        let sampler = DistributedSampler::with_world(
            world.clone(),
            w / workers_per_rank,
            w % workers_per_rank,
            global_seed,
            algo,
        )
        .with_shard_order(shard_order);
        let mut seq: Vec<u32> = Vec::new();
        for epoch in 0..epochs {
            let indices = sampler.my_shard_indices(num_shards, epoch);
            // 本 worker 本 epoch 的 batch 前缀和：prefix[k] = (累计 batch 数, 第 k 个 shard 的全局下标)
            let mut prefix: Vec<(u64, usize)> = Vec::with_capacity(indices.len());
            let mut acc: u64 = 0;
            for &gi in &indices {
                acc += manifest.shards[gi].num_batches as u64;
                prefix.push((acc, gi));
            }
            let total = acc;
            let perm = if shuffle {
                batch_permutation(total, global_seed, epoch)
            } else {
                (0..total as u32).collect()
            };
            for p in perm {
                let p = p as u64;
                // 在前缀和里定位 p 落在哪个 shard。语义与 SyncInnerReader::position_target 一致：
                // 第 k 个 shard 覆盖 [prefix[k-1].0, prefix[k].0)。二分（partition_point）而非线性扫描：
                // 仿真器可能跑几十万 step × 几百 shard。
                let k = prefix.partition_point(|&(end, _)| end <= p);
                let shard_gi = prefix.get(k).map(|&(_, gi)| gi).unwrap_or(0);
                seq.push(shard_gi as u32);
            }
        }
        out.push(seq);
    }
    out
}

/// batch 置换表（置换表 shuffle，替代 Python 端有状态 buffer shuffle）：
/// 输入 (n, seed, epoch)，输出 0..n 的确定性置换。恢复训练只需记录
/// "epoch + epoch 内位置"，checkpoint 从 GB 级 buffer 缩到几十字节。
///
/// n 超过 u32::MAX 时返回**空表**——注意调用方（`position_target`）查空表
/// 返回 None，`resolve_next_position` 会将其当作 internal state inconsistency
/// **报错**，并不会退回顺序读取。batch 级置换表长度 = batch 数，比 sample 级
/// 小 batch_size 倍，正常规模远达不到该上限；若未来确需支持超大 epoch，
/// 应在 StreamReader 层显式实现顺序回退。
pub fn batch_permutation(n: u64, seed: u64, epoch: u64) -> Vec<u32> {
    if n == 0 || n > u32::MAX as u64 {
        return Vec::new();
    }
    let mut rng = StdRng::seed_from_u64(seed ^ epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.shuffle(&mut rng);
    idx
}

/// 记录当前数据流读取进度的快照（checkpoint schema 记录，保留为稳定 API）
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamState {
    pub epoch: u64,
    pub shard_list_index: usize, // 当前读到了分配给我的 Shard 列表中的第几个
    pub batches_yielded_in_shard: u64, // 当前这个 Shard 里已经吐出了多少个 Batch

    // ===== 方案 3 新增字段 (向后兼容：用 #[serde(default)] 容忍老 checkpoint) =====
    /// 已经从本 rank 数据集中消费过的总 batch 数（跨 epoch 累加）。
    /// MapDataset 模式用于支持 __getitem__(idx) 的精确 seek。
    /// 流式模式同样用于判断"消费到第几个 batch"以决定是否到达 num_epochs 上限。
    #[serde(default)]
    pub global_batch_idx: u64,

    /// 用户设置的总 epoch 数（0 = 永远不停止流式读取，即原"流式单 epoch"行为）。
    #[serde(default)]
    pub num_epochs: u64,

    // ===== 弹性扩缩容锚点（relaxed 分区，借鉴 streaming 五元组 checkpoint）=====
    /// checkpoint 时的规范节点数：续训时传入，保证全局样本序与原 run 一致。
    #[serde(default)]
    pub num_canonical_nodes: Option<usize>,
    /// checkpoint 时的物理节点数（streaming 五元组里的 `initial_physical_nodes`）。
    ///
    /// 注意：本仓库的弹性锚点是 `num_canonical_nodes`（见 `relaxed_epoch_order`）——
    /// relaxed 分区的全局样本序只由"规范节点数"决定，换物理节点数续训不需要再记它。
    /// 该字段保留为 checkpoint schema（与 streaming / 旧 checkpoint 对齐），
    /// 导出快照时填 `World.num_nodes` 即可；当前没有运行时构造点，故恒为 None。
    #[serde(default)]
    pub initial_physical_nodes: Option<usize>,
}

#[allow(dead_code)]
impl StreamState {
    pub fn new(epoch: u64) -> Self {
        Self {
            epoch,
            shard_list_index: 0,
            batches_yielded_in_shard: 0,
            global_batch_idx: 0,
            num_epochs: 0,
            num_canonical_nodes: None,
            initial_physical_nodes: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_shard_sequence_consistent_and_covering() {
        use crate::builder::metadata::{DatasetManifest, ShardMeta, StorageFormat};
        let mut manifest = DatasetManifest::new(StorageFormat::Memmap, 128);
        for i in 0..12 {
            manifest.shards.push(ShardMeta {
                shard_id: i,
                filename: format!("s{i}.bin"),
                num_batches: 2 + (i % 3),
                size_bytes: 1024,
                hashes: Default::default(),
                align_aud_range: None,
            });
        }
        let world = World::from_sizes(4, 1, 2); // 4 rank × 2 worker = 8 workers
        let total_batches: u64 = manifest.shards.iter().map(|s| s.num_batches as u64).sum();

        // shuffle 模式：每 epoch 内 batch 序列是全域置换，8 个 worker 的步数总和应精确守恒
        let plan = plan_shard_sequence(
            &manifest,
            &world,
            PartitionAlgo::RoundRobin,
            9,
            2,
            true,
            ShardOrder::Random,
        );
        assert_eq!(plan.len(), 8);
        let total_steps: u64 = plan.iter().map(|w| w.len() as u64).sum();
        assert_eq!(
            total_steps,
            total_batches * 2,
            "total worker steps should == total_batches × epochs"
        );
        for w in &plan {
            assert!(
                !w.is_empty(),
                "12 shards >= 8 workers: each worker should get at least 1 shard"
            );
            assert!(w.iter().all(|&s| (s as usize) < 12));
        }
        // 每 worker 每 epoch 的 shard 多重集 == 其分区内各 shard 的 num_batches 展开
        for (w, seq) in plan.iter().enumerate() {
            let sampler = DistributedSampler::with_world(
                world.clone(),
                w / 2,
                w % 2,
                9,
                PartitionAlgo::RoundRobin,
            );
            let mut expected: Vec<u32> = Vec::new();
            for epoch in 0..2u64 {
                for gi in sampler.my_shard_indices(12, epoch) {
                    expected.extend(std::iter::repeat_n(
                        gi as u32,
                        manifest.shards[gi].num_batches,
                    ));
                }
            }
            let mut got = seq.clone();
            got.sort_unstable();
            expected.sort_unstable();
            assert_eq!(
                got, expected,
                "worker {w}'s shard multiset should match the partition expansion"
            );
        }

        // 非 shuffle 模式（epochs=1）：worker w 的序列 = 其 shard 列表按 num_batches 顺序展开
        let plan_seq = plan_shard_sequence(
            &manifest,
            &world,
            PartitionAlgo::RoundRobin,
            9,
            1,
            false,
            ShardOrder::Random,
        );
        let expected_w0: Vec<u32> = {
            let s =
                DistributedSampler::with_world(world.clone(), 0, 0, 9, PartitionAlgo::RoundRobin);
            s.my_shard_indices(12, 0)
                .into_iter()
                .flat_map(|gi| std::iter::repeat_n(gi as u32, manifest.shards[gi].num_batches))
                .collect()
        };
        assert_eq!(
            plan_seq[0], expected_w0,
            "non-shuffle sequence should match the shard list expansion in order"
        );
        let total_steps: u64 = plan_seq.iter().map(|w| w.len() as u64).sum();
        assert_eq!(total_steps, total_batches);
    }

    fn sampler(
        algo: PartitionAlgo,
        world_size: usize,
        rank: usize,
        worker: usize,
        workers: usize,
    ) -> DistributedSampler {
        let world = World::from_sizes(world_size, 1, workers);
        DistributedSampler::with_world(world, rank, worker, 42, algo)
    }

    fn assert_disjoint_and_cover(sampler_sets: Vec<Vec<usize>>, n: usize) {
        let mut seen = vec![false; n];
        for set in &sampler_sets {
            for &i in set {
                assert!(!seen[i], "shard {i} was assigned more than once");
                seen[i] = true;
            }
        }
        assert!(seen.iter().all(|&s| s), "some shards were never assigned");
    }

    #[test]
    fn test_round_robin_disjoint_and_cover() {
        let n = 100;
        let sets: Vec<Vec<usize>> = (0..4)
            .map(|r| sampler(PartitionAlgo::RoundRobin, 4, r, 0, 1).my_shard_indices(n, 0))
            .collect();
        assert_disjoint_and_cover(sets, n);
    }

    #[test]
    fn test_worker_dimension_disjoint_and_cover() {
        // 2 rank × 4 worker：8 个 (rank, worker) 必须互斥覆盖全集
        let n = 64;
        let sets: Vec<Vec<usize>> = (0..2)
            .flat_map(|r| (0..4).map(move |w| (r, w)))
            .map(|(r, w)| sampler(PartitionAlgo::RoundRobin, 2, r, w, 4).my_shard_indices(n, 7))
            .collect();
        assert_disjoint_and_cover(sets, n);
    }

    #[test]
    fn test_epoch_reshuffle_changes_assignment() {
        let s = sampler(PartitionAlgo::RoundRobin, 4, 0, 0, 1);
        let e0 = s.my_shard_indices(100, 0);
        let e1 = s.my_shard_indices(100, 1);
        assert_ne!(
            e0, e1,
            "shard assignment should differ across epochs (epoch reshuffle)"
        );
    }

    // ---- F6：shard_order="sequential" ----

    #[test]
    fn test_shard_order_sequential_round_robin_is_identity() {
        // F6 验收：sequential 下 RoundRobin 全局序恒为 0..n，且与 epoch 无关
        let s =
            sampler(PartitionAlgo::RoundRobin, 1, 0, 0, 1).with_shard_order(ShardOrder::Sequential);
        let expected: Vec<usize> = (0..100).collect();
        for epoch in 0..4u64 {
            assert_eq!(
                s.my_shard_indices(100, epoch),
                expected,
                "sequential order must be the identity regardless of epoch"
            );
        }
        // 默认（random）跨 epoch 仍打散（行为不变）
        let r = sampler(PartitionAlgo::RoundRobin, 1, 0, 0, 1);
        assert_ne!(r.my_shard_indices(100, 0), r.my_shard_indices(100, 1));
        // 且默认行为与修复前一致：random 不等于 identity（100 个元素打散后极不可能恒等）
        assert_ne!(r.my_shard_indices(100, 0), expected);
    }

    #[test]
    fn test_shard_order_sequential_multi_worker_disjoint_and_cover() {
        // sequential + 多 worker：仍互斥覆盖全集（只是顺序变成按编号轮询）
        let n = 37;
        let sets: Vec<Vec<usize>> = (0..4)
            .map(|r| {
                sampler(PartitionAlgo::RoundRobin, 4, r, 0, 1)
                    .with_shard_order(ShardOrder::Sequential)
                    .my_shard_indices(n, 2)
            })
            .collect();
        assert_disjoint_and_cover(sets.clone(), n);
        // worker 0 依次拿 0, 4, 8, ...（按编号轮询，无 epoch 打散）
        assert_eq!(sets[0], vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36]);
    }

    #[test]
    fn test_shard_order_sequential_relaxed_stable_across_epochs() {
        // relaxed + sequential：全局序只由 (seed, ncn) 决定，跨 epoch 稳定，
        // 且与 relaxed_epoch_order 的成员集合一致（同一 run_rng）
        let n = 50;
        let s =
            sampler(PartitionAlgo::Relaxed, 1, 0, 0, 1).with_shard_order(ShardOrder::Sequential);
        let e0 = s.my_shard_indices(n, 0);
        let e1 = s.my_shard_indices(n, 1);
        assert_eq!(
            e0, e1,
            "sequential relaxed order must be stable across epochs"
        );
        let mut sorted = e0.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..n).collect::<Vec<usize>>(),
            "must be a permutation of 0..n"
        );
        // 与 random 变体互为排列（成员集合相同，顺序不同）
        let shuffled = relaxed_epoch_order(n, 42, 0, 1);
        let mut a = e0.clone();
        let mut b = shuffled.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "same shard multiset as the random variant");
        assert_ne!(
            e0, shuffled,
            "sequential skips the within-block epoch shuffle"
        );
    }

    #[test]
    fn test_relaxed_sequential_order_matches_canonical_blocks() {
        // ncn=4：块内按序、块间按 node 编号；成员 = relaxed_epoch_order 的对应块成员
        let n = 40;
        let seq = relaxed_sequential_order(n, 7, 4);
        let mut run_rng = StdRng::seed_from_u64(7);
        let canon: Vec<usize> = (0..n).map(|_| run_rng.random_range(0..4)).collect();
        let mut expected: Vec<usize> = Vec::new();
        for node in 0..4 {
            expected.extend((0..n).filter(|&i| canon[i] == node));
        }
        assert_eq!(
            seq, expected,
            "sequential relaxed order = canonical blocks in index order"
        );
        let shuffled = relaxed_epoch_order(n, 7, 0, 4);
        for node in 0..4 {
            let mut in_seq: Vec<usize> =
                seq.iter().copied().filter(|&s| canon[s] == node).collect();
            let mut in_shuffled: Vec<usize> = shuffled
                .iter()
                .copied()
                .filter(|&s| canon[s] == node)
                .collect();
            in_seq.sort_unstable();
            in_shuffled.sort_unstable();
            assert_eq!(
                in_seq, in_shuffled,
                "block membership must match relaxed_epoch_order"
            );
        }
    }

    #[test]
    fn test_shard_order_parse() {
        assert_eq!(ShardOrder::parse("random").unwrap(), ShardOrder::Random);
        assert_eq!(
            ShardOrder::parse("sequential").unwrap(),
            ShardOrder::Sequential
        );
        assert!(ShardOrder::parse("sorted").is_err());
    }

    #[test]
    fn test_plan_shard_sequence_sequential_expands_in_order() {
        // sequential + shuffle=false：worker 0 的 step 序列 = shard 0..n 各自的
        // num_batches 顺序展开（离线工具可逐位复现）
        use crate::builder::metadata::{DatasetManifest, ShardMeta, StorageFormat};
        let mut manifest = DatasetManifest::new(StorageFormat::Memmap, 128);
        for i in 0..6 {
            manifest.shards.push(ShardMeta {
                shard_id: i,
                filename: format!("s{i}.bin"),
                num_batches: 2 + (i % 3),
                size_bytes: 1024,
                hashes: Default::default(),
                align_aud_range: None,
            });
        }
        let world = World::from_sizes(1, 1, 0);
        let plan = plan_shard_sequence(
            &manifest,
            &world,
            PartitionAlgo::RoundRobin,
            9,
            3,
            false,
            ShardOrder::Sequential,
        );
        assert_eq!(plan.len(), 1);
        let mut expected: Vec<u32> = Vec::new();
        for _ in 0..3 {
            for (gi, s) in manifest.shards.iter().enumerate() {
                expected.extend(std::iter::repeat_n(gi as u32, s.num_batches));
            }
        }
        assert_eq!(
            plan[0], expected,
            "sequential non-shuffle plan = per-shard batches in shard order"
        );
    }

    #[test]
    fn test_relaxed_disjoint_and_cover() {
        let n = 97; // 非整除，考察余数分配
        let sets: Vec<Vec<usize>> = (0..5)
            .map(|r| sampler(PartitionAlgo::Relaxed, 5, r, 0, 1).my_shard_indices(n, 3))
            .collect();
        assert_disjoint_and_cover(sets.clone(), n);
        // 均衡性：|len_i - len_j| <= 1
        let lens: Vec<usize> = sets.iter().map(|s| s.len()).collect();
        assert!(lens.iter().max().unwrap() - lens.iter().min().unwrap() <= 1);
    }

    #[test]
    fn test_relaxed_order_topology_independent() {
        // 全局序只依赖 (seed, epoch, ncn)：不同 world_size 下的"拼接序"一致，
        // 这是跨拓扑续训的根基。
        let n = 50;
        let order1 = relaxed_epoch_order(n, 42, 2, 3);
        let order2 = relaxed_epoch_order(n, 42, 2, 3);
        assert_eq!(order1, order2);

        // 两个 world_size 下各 worker 切片的按序拼接 == 全局序的一个排列
        let s2: Vec<Vec<usize>> = (0..2)
            .map(|r| sampler(PartitionAlgo::Relaxed, 2, r, 0, 1).my_shard_indices(n, 2))
            .collect();
        let s4: Vec<Vec<usize>> = (0..4)
            .map(|r| sampler(PartitionAlgo::Relaxed, 4, r, 0, 1).my_shard_indices(n, 2))
            .collect();
        let concat2: Vec<usize> = s2.into_iter().flatten().collect();
        let concat4: Vec<usize> = s4.into_iter().flatten().collect();
        let sorted = |mut v: Vec<usize>| {
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(concat2.clone()), sorted(order1.clone()));
        assert_eq!(sorted(concat4), sorted(order1));
    }

    #[test]
    fn test_relaxed_download_locality_across_epochs() {
        // 同一 shard 在不同 epoch 的归属 canonical node 稳定（块内打散），
        // 而纯轮询/全局打散做不到这一点。
        let n = 100usize;
        let (o0, o1) = (
            relaxed_epoch_order(n, 7, 0, 4),
            relaxed_epoch_order(n, 7, 1, 4),
        );
        // 重建 run_rng 的 canonical node 归属（与实现一致），按归属分块比较
        let mut run_rng = StdRng::seed_from_u64(7);
        let canon: Vec<usize> = (0..n).map(|_| run_rng.random_range(0..4)).collect();
        for node in 0..4 {
            let in_block = |o: &[usize]| -> Vec<usize> {
                let mut v: Vec<usize> = o.iter().copied().filter(|&s| canon[s] == node).collect();
                v.sort_unstable();
                v
            };
            assert_eq!(
                in_block(&o0),
                in_block(&o1),
                "members of node {node} block should be stable across epochs"
            );
        }
        // 但 epoch 内顺序不同（块内打散）
        assert_ne!(o0, o1);
        // 且都是双射
        let sorted = |o: &[usize]| {
            let mut v = o.to_vec();
            v.sort_unstable();
            v
        };
        assert_eq!(sorted(&o0), sorted(&o1));
        assert_eq!(sorted(&o0), (0..n).collect::<Vec<usize>>());
    }

    #[test]
    fn test_batch_permutation_deterministic_and_bijective() {
        let p1 = batch_permutation(1000, 9, 0);
        let p2 = batch_permutation(1000, 9, 0);
        assert_eq!(p1, p2);
        let mut sorted: Vec<u32> = p1.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            (0..1000).collect::<Vec<u32>>(),
            "permutation must be a bijection"
        );
        let p3 = batch_permutation(1000, 9, 1);
        assert_ne!(p1, p3, "permutation should differ across epochs");
        assert!(batch_permutation(0, 9, 0).is_empty());
    }

    #[test]
    fn test_contiguous_slice_bounds() {
        // n=10, W=4: [0,3)(3个) [3,6)(3个) [6,8)(2个) [8,10)(2个)
        assert_eq!(contiguous_slice_bounds(10, 0, 4), (0, 3));
        assert_eq!(contiguous_slice_bounds(10, 1, 4), (3, 3));
        assert_eq!(contiguous_slice_bounds(10, 2, 4), (6, 2));
        assert_eq!(contiguous_slice_bounds(10, 3, 4), (8, 2));
        // worker 超出 shard 数：空切片
        assert_eq!(contiguous_slice_bounds(2, 5, 8), (2, 0));
    }

    #[test]
    fn test_world_sizes() {
        let w = World::from_sizes(16, 2, 4);
        assert_eq!(w.world_size(), 16);
        assert_eq!(w.total_workers(), 64);
        assert_eq!(w.canonical_nodes(), 2);
        let mut w2 = w.clone();
        w2.num_canonical_nodes = Some(8);
        assert_eq!(w2.canonical_nodes(), 8);
    }
}
