# diskfm — 外存子串索引（磁盘 FM-index）

对任意 **Unicode 精确子串** 建索引并查询重叠出现次数与坐标。查询不是词项、
不是前缀、不是相似词：后向搜索（backward search）直接在完整 BWT 上匹配任意
码点序列。构建与查询的工作内存都不随语料增长，文本总量可以大于内存。

- 只使用 Python 标准库；**不调用**任何现成后缀数组 / FM-index / 全文检索库。
- 排序键只含整数（位置、rank、行号），**从不把完整后缀复制成排序键**。
- rank 结构真实落盘；查询时不读完整 SA、不解码全部原文、不做全文扫描。

## 快速开始

```bash
# 构建（chunk-items 是唯一的内存旋钮：内存排序缓冲区最多容纳多少条记录）
python3 -m diskfm build input.jsonl my.idx \
    --chunk-items 20000 --sample-interval 32

python3 -m diskfm count  my.idx '任意子串'
python3 -m diskfm locate my.idx '任意子串' --page-size 64 --all
python3 -m diskfm verify my.idx      # 重新校验全部哈希与几何
python3 -m diskfm stats  my.idx      # 清单 + 实测 rank/LF/分页计数
python3 -m diskfm get    my.idx doc_id   # 逐字节取回原始文档
```

输入 JSONL：每行一个对象，含唯一字符串 `doc_id` 与字符串 `text`，空行允许。
原文逐字节保留（`content.bin`），不分词、不正规化、不折叠大小写；允许空文档、
换行、NUL；**拒绝孤立代理码点**（无法作为 Unicode 文本合法表示）。

## 排序次序（locate 的确定性顺序）

全局符号流

```
T = doc0文本 + S0 + doc1文本 + S1 + ... + Sd + Sd
```

其中内部文档终点符 `Sd = 0x110000 + d`。后缀次序按整数符号字典序确定：

1. 合法用户码点按其精确码点值比较（`0 .. 0x10FFFF`），原样比较；
2. 所有内部符号严格大于任何合法用户码点（最小内部符号 `0x110000`，
   与 Unicode 最大合法码点 `U+10FFFF` 不冲突，也不可能被用户查询包含）；
3. 内部符号之间按文档下标递增排序，因此每个文档终点可唯一区分。

`locate` 的返回顺序就是该后缀次序中匹配区间的行顺序（不是文档输入序）。
每个文档以唯一哨兵结束，而任何查询都不可能含 `>= 0x110000` 的符号，所以
**匹配区间不可能跨越文档边界**；相同文本的不同 `doc_id` 是不同后缀，分别计数。
“文档 ab + 文档 a” 不会拼出 `aba`。

## 构建路线（全部外存，可解释）

1. **摄入**：单次流式读 JSONL。码点写 `text.bin`（u32 序列），原始 UTF-8 字节
   逐字节写 `content.bin`，同时写文档坐标表与字节偏移采样。
2. **前缀翻倍（prefix doubling）**，每轮至多两次外存归并排序：
   - 第 0 轮外排 `(T[p], p)` 得到长度 1 的稠密 rank；
   - 第 k 轮外排 `(rank[p], rank[(p+k) mod N], p)`，重新编号不同 rank 对，
     再外排 `(p, new_rank)` 恢复按位置索引的 rank 文件；
   - 一旦 N 个 rank 互不相同，排序结果本身就是后缀数组，提前停止
     （重复文本上会明显提前，manifest 的 `early_stop` 记录此事）。
   - 记录用定长编码：能压进 64 位时用 8 字节打包键（`Packed64Codec`），
     位宽不够时用 12 字节三 u32 宽记录（`Triple32Codec`，`--force-wide`
     可强制走这条路径测试）。
3. **BWT 与 SA 采样**：流式读 SA，按循环规则
   `BWT[row] = T[(SA[row]-1) mod N]` 输出 BWT；每隔 `sample_interval` 行
   存一个 `(row, T位置)` 采样。
4. **落盘 rank**：把 `(BWT符号, 行号)` 对再做一次外存归并排序，得到每个符号
   的有序出现行号表（`rank.bin` + 目录 `rankdir.bin`）。
   `rank(c, i)` = 在 c 的出现表中对 i 做分页二分，O(log 出现次数) 次磁盘读，
   不是计数数组、不是全文扫描。

### 预算控制

- 内存排序缓冲区至多 `chunk_items` 条记录，满了就排序落盘成一个 run；
- k 路归并的堆每路只放一条头记录，归并 RAM 为 O(路数) 而非 O(语料)；
- 所有随机访问（构建期 rank 文件；查询期 rank 出现表、rank 符号目录、
  SA 采样表、BWT、字节检查点、文档坐标表与 doc_id 块）都经过有界 LRU
  分页器 `pager.FilePager`：固定 `max_pages` 个页，驻留字节数与 N / 文档数
  无关；打开索引时对 rankdir、doctable 的完整性校验本身也是分页流式进行，
  不把这两个结构整体读进内存；
- 没有任何步骤把后缀字符串当排序键，也没有在内存里物化完整 SA。

## 查询：count 与 locate

- **count**：从空区间 `[0,N)` 开始，对查询**从右向左**逐个码点做
  `lo = C(c)+rank(c,lo)`、`hi = C(c)+rank(c,hi)`。结果就是区间宽度，
  不枚举任何位置；count 路径上 LF 步数、SA 采样读取、原文读取均为 0。
- **locate（分页）**：对区间内本页的每一行做坐标恢复，页外命中不恢复：
  - 从行 `cur` 反复 LF：`LF(i)=C(BWT[i])+rank(BWT[i],i)`，直到落到采样行；
    不变式为 t 步后 `SA[cur_t] = (p0 - t) mod N`，故 `p0 = 采样位置 + t`。
  - **跨界处理**：LF 在循环符号流上良定义，即使穿过哨兵也保持精确；实现里
    做了一步提前优化——若 `BWT[cur_t]` 是文档 j 的哨兵，则必有
    `SA[cur_t] = 哨兵位置(j)+1`，于是 `p0 = (哨兵位置(j)+1+t) mod N`，
    根本不执行跨越文档的 LF 步，也不会制造跨文档命中。
  - 码点偏移 = `p0 - 文档T起点`；UTF-8 字节偏移由每文档的字节检查点
    （`bytesample.bin`）加增量 UTF-8 解码得到，窗口不进入下一文档。
- 页内不保留全部命中；下一页由不透明游标续读。

### 游标绑定

游标令牌内含并校验：索引实例身份（每个 build 生成的 UUID，内容相同的两个
索引身份也不同）、查询片段的 SHA-256、当前匹配区间 `[lo,hi)`、页内偏移，
外加载荷自身的 SHA-256 摘要防篡改。拿到另一索引或另一查询上续读会抛
`CursorError`；`None` 游标表示从头开始。

## 磁盘格式（版本 1）

每个数据文件以 8 字节头开始：`magic=0x464D, version, kind, reserved`。

| 文件 | 内容 |
| --- | ---|
| `manifest.json` | 格式版本、实例身份、内容哈希、几何、构建统计、每个文件的 SHA-256+长度 |
| `OPEN` | 完成标志（最后写入并 fsync）；缺失即“构建中断”，索引拒绝打开 |
| `text.bin` | 全局 u32 符号流（用户码点 + 内部哨兵） |
| `content.bin` | 原始 UTF-8 字节，文档首尾相接，NUL/换行原样 |
| `bytesample.bin` | 每文档的 UTF-8 字节偏移检查点（u64） |
| `doctable.bin` | 定长 48B 文档记录（含 doc_id 块偏移与检查点下标）+ doc_id 字节块，分页访问 |
| `bwt.bin` | 每行一个 u32 BWT 符号 |
| `rank.bin` / `rankdir.bin` | 每符号的有序出现行号表，及其分页目录（含 C(c) 累计值） |
| `sa_sample.bin` | 采样 `(u32 行, u64 T位置)` |

打开时校验：OPEN 标志、magic/version/kind、manifest 与每个文件的长度和
SHA-256、rank 目录总数等于 N、采样行可查、doctable 无残缺或尾随垃圾。
任何截断 / 越界 / 损坏都抛 `IndexCorruptError`，而不是少报匹配。

## 实测计数器（源码内保留）

`DiskFMIndex.stats()` 与 `python3 -m diskfm stats` 暴露：`rank_calls`、
rank / rankdir / doctable / BWT / 采样各分页器的命中、未命中与驻留字节、
`lf_steps`、采样行读取数、字节检查点读取、原文读取字节数。构建侧
`manifest.build_stats` 记录翻倍轮数、外排次数、落盘 chunk 数与字节、
实际用到的最宽记录编码。

## 测试

```bash
cd tests
python3 -I pytest.py test_diskfm.py   # 49 项差分/完整性/分页测试（无第三方依赖）
python3 -I pytest.py test_large.py    # 大语料有界内存证据 + CLI 端到端
```

差分测试的期望结果由 `tests/oracle.py` 独立生成：对**原文**反复
`str.find(query, 上一次起点+1)` 统计重叠命中，并用 UTF-8 前缀长度独立计算
两种坐标，完全不接触 BWT/rank 代码。覆盖：`ababa` 中 `aba` 的 {0,2}、
文档不跨界、emoji/NUL 与字节偏移、最大合法码点与哨兵不冲突、孤立代理拒绝、
长重复串、相同文档分别计数、空文档、多次 build 输入顺序扰动、页长 1 取完、
游标换查询 / 换索引 / 篡改拒绝、采样间隔边界值、截断 / 损坏 / 中断失败等。
