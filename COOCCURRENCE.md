# diskfm 共现分析 —— 多短语窗口共现计数与位置对

在已封存的 diskfm 索引之上，回答“给定短语集合与短语对，各短语对在
同一文档中、码点距离不超过 `window` 的命中位置对有多少、分别在哪里”，
并按文档给出可复核的位置对分页。

这是一项**短语关系分析**：不是把近重复文档再分组（那是
[SIMILARITY.md](SIMILARITY.md) 的 Jaccard 目录），也不是先逐短语调用
`locate` 再在内存里拼接结果。它直接建立在 FM-index 的命中区间之上，
把按后缀序返回的命中转成可连接的位置流，再做精确窗口连接。

## 语义

- 短语：非空、原始 Unicode 字符串，按索引中存储的码点序列**精确匹配**；
  不分词、不正规化、不折叠大小写；命中允许重叠。
- 每个短语有一个唯一 `pattern_id`。不同 `pattern_id` 即使文本相同，仍是
  不同分析项；同一 `pattern_id` 不得与自身配对。
- 对请求中的短语对 `(A, B)`：在同一 `doc_id` 内各取一个 A 命中（起点
  码点偏移 `a`）与一个 B 命中（起点偏移 `b`），当且仅当
  `abs(a - b) <= window` 时构成一个**共现位置对**。每个不同的命中位置
  组合只计一次。`(A@p, B@q)` 与 `(B@q, A@p)` 是两个不同的分析项。
- **不跨文档配对**；零共现文档不输出。
- 输出位置对保留请求的双方：`a_codepoint_offset`/`b_codepoint_offset`
  与 `a_utf8_byte_offset`/`b_utf8_byte_offset`。

## 伪代码

```python
from diskfm import DiskFMIndex

with DiskFMIndex("my.idx") as idx:
    analysis = idx.cooccur(
        patterns=[("A", "the quick"), ("B", "brown fox"), ("C", "fox")],
        pairs=[("A", "B"), ("A", "C"), ("B", "C")],
        window=12,
    )
    for row in analysis.iter_summary():
        print(row)          # 每 (pair, doc) 的非零总数，按请求对序 + doc_id
    for row in analysis.iter_pairs(("A", "B")):
        print(row)          # 每对位置对，含两种偏移
    rows, cursor = analysis.pairs_page(("A", "B"), page_size=100)
    rows, cursor = analysis.pairs_page(("A", "B"), page_size=100, cursor)
```

命令行：

```bash
python3 -m diskfm.cooccurcli run my.idx request.json --all
python3 -m diskfm.cooccurcli run my.idx request.json \
    --pair A B --pair-page-size 100
```

`request.json` 形如：

```json
{"patterns": [{"pattern_id": "A", "phrase": "the quick"},
              {"pattern_id": "B", "phrase": "brown fox"}],
 "pairs": [["A", "B"]],
 "window": 12}
```

## 为什么后缀序不能直接连接

`locate` 按**后缀序**返回命中行。后缀序按后缀比较交错文档，既不是
`doc_id` 序，也不是文档内位置序；两个短语的命中流在后缀序下无法直接
做窗口连接，分页也不能改变这一点。

本实现因此把每个短语的命中转成可连接的位置流：

1. `DiskFMIndex.iter_raw_hits` 对短语的后向搜索区间**恰好走一遍**
   （无游标、无分页接缝），产出 `(doc_index, codepoint_offset)`；行数
   就是精确的重叠命中数，热门短语的命中不会被截断。
2. 这些命中进入**有界外排**，键是完整记录 `(doc_index, codepoint_offset)`，
   写出定长 `hits.bin` 与逐文档分组表。键不是哈希、不是截断前缀，因此
   不同命中永远不会比较相等；同短语的命中严格有序。
3. 分组表按文档序 k 路归并；对每个文档、每个请求短语对，用**双指针
   扫描**两段有序位置。闭区间 `[a-window, a+window]` 恰好覆盖
   `abs(a-b) <= window`，每个组合只生成一次。扫描只用 O(1) 状态计数，
   位置对流式送入第二个有界外排（按“请求对序 → doc_id 码点序 →
   (a_cp, b_cp)”排序）。

因为连接以文档为单位、基于完整位置键、边界是闭区间，所以位置对在
`window` 恰等于差值、`window == 0`（同偏移）、相邻/重叠命中，以及两条
溢写 run 的接缝处都**恰好生成一次**。

## 明细：恰好一次与不丢命中

- 命中来源是对区间逐行遍历，不存在分页边界，因此不会因分页/外排分段
  重复或遗漏。
- 外排键包含完整 `(doc, cp)`；不同命中不会碰撞。
- 连接在每个文档内对两段有序切片做单调双指针；对每个 A 命中，`j_lo`
  只能前进到 `>= a-window`，`j_hi` 只能前进到 `> a+window`，所以每个
  `(a, b)` 组合恰好在“b 位于当前 A 的窗口内”时被发射一次。
- 汇总计数 `candidate_pairs` 与最终落盘的位置对总数、以及 summary
  逐行计数三者在构建末尾交叉校验；不一致即拒绝发布。

## 位置与字节坐标

码点偏移由 FM 的 SA 采样 + LF 还原得到；UTF-8 字节偏移由每个文档的
字节检查点推导。实现为每个文档、每个短语开启一个**惰性前向解码游标**
（`DocumentByteCursor`）：只按需推进到下一个命中偏移，持有 1 个增量
解码器与两个整数——不缓冲文档，不逐命中做前缀重算，因此内存与命中
密度无关，且每个文档的字节最多被线性解码一遍。

## 校验、身份与分页

- 入口校验请求结构（`pattern_id` 非空且唯一、短语非空且无孤立代理、
  短语对两端已知且不相同、无重复对、窗口为非负整数且不越界）以及
  索引完成状态/身份（打开索引时已校验 OPEN、manifest 哈希与实例 UUID）。
- 分析结果以 `(索引实例 UUID, 请求摘要)` 为键发布到索引目录**之外**
  的缓存目录（`DISKFM_COOCCUR_CACHE` 或系统临时目录），原子发布
  （staging → fsync OPEN → rename）；不会修改已封存索引。
- 游标绑定：索引实例身份、完整规范化请求摘要（含短语 id 顺序、短语
  文本、短语对顺序、窗口）、分析摘要、结果部分与位置。换索引、改任一
  分析条件、换短语对或篡改都会被 `CursorError` 拒绝；损坏的结果文件
  抛 `IndexCorruptError`。
- 结果可分页完整取尽；相同索引与请求的重复执行产生相同逻辑结果，且
  游标可在另一个进程中续读（结果可确定性重算或从缓存重开）。

## 内存与磁盘边界

- 每个外排缓冲区最多 `chunk_items` 条记录 / `chunk_bytes` 字节，先到先
  溢写；归并每路只放一条头记录，扇入受 `merge_fan_in` 限制。
- 逐文档连接计数只用 O(1) 状态；位置对流式外排，不把任何文档的配对
  整体放入内存。
- 命中分组与汇总的流式读取、位置对分页都基于定长/带偏移索引的文件，
  经固定页数 LRU 读取，驻留字节与结果规模无关。
- 所有命中可达，无截断/近似；计数精确。

## 统计（均来自执行，不估算）

`CooccurAnalysis.stats()["analysis"]`：`hits_read`、`hits_per_phrase`、
`hit_sort_runs`/`hit_sort_bytes`/`hit_sort_merge_passes`、
`documents_with_hits`、`phrase_document_groups`、`candidate_pairs`、
`position_pairs`、`pair_counts`、`summary_rows`、
`pair_sort_runs`/`pair_sort_bytes`、`summary_sort_*`、`hit_file_bytes`、
`result_file_bytes`、`temp_disk_bytes`、`content_sha256`。
`stats()["paging"]`：summary/pairs 页数、读取记录数、分页器命中/未命中/
驻留字节。峰值 RSS 由 `scripts/cooccur_mem_worker.py` 从
`/proc/self/status` 读取。

## 测试

```bash
cd tests
python3 -I pytest.py test_cooccur.py       # 23 项差分/分页/游标/损坏/失败隔离
python3 -I pytest.py test_cooccur_large.py # 大语料有界内存与密集命中证据
```

期望结果由 `tests/cooccur_oracle.py` 独立生成：直接在原文逐位置枚举每个
短语命中，再独立枚举指定短语对与同文档位置对，按
`abs(a-b) <= window` 计算计数与两种坐标，完全不接触生产搜索、排序或
窗口连接代码。覆盖：相同短语不同 ID、重叠/相邻命中、窗口恰等于差值及
差一位、两种短语先后颠倒、同文档多次出现、跨文档不得连接、无命中、
高频短语密集配对、`window=0`、长短语、emoji/NUL/换行、摘要计数、
位置对分页、改请求/换索引后的游标拒绝、外排分段边界、临时空间耗尽、
截断与损坏。测试数据运行时生成，仓库只保留小型可读夹具。
