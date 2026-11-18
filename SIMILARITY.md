# diskfm 相似度目录 —— 文档级近重复审计

在已封存的 diskfm 文本索引之上，为同一批文档构建**独立、可重开**的相似度
目录（similarity catalog），回答：哪些文档是近重复、重复了多少（Jaccard）、
应先人工合并哪几对（按相似度降序的稳定分页），并能解释任意两篇文档共享的
具体 shingle。目录只存 qgram 倒排关系，**不存原文**；查询路径没有任何办法
回读全体原文，也不会对任何文档重新生成 shingle。

只使用 Python 标准库；核心处理是磁盘倒排 + 有界内存外部归并排序，不是
“对每个文档逐一调用既有 substring 查询”的脚本，也不是查询时全文扫描。

## 快速开始

```bash
# 先有一份已封存的 diskfm 索引（见 README.md）
python3 -m diskfm build input.jsonl my.idx

# 构建相似度目录（q 为 shingle 长度，单位是 Unicode 码点）
python3 -m diskfm.simcli build my.idx my.sim --q 5 \
    --chunk-items 20000 --chunk-bytes 4194304

# 查询某篇文档的近重复（稳定分页；threshold 支持小数或 num/den）
python3 -m diskfm.simcli query my.sim doc_017 --threshold 0.8 --page-size 20
python3 -m diskfm.simcli query my.sim doc_017 --threshold 0.8 --all
python3 -m diskfm.simcli query my.sim doc_017 --threshold 3/4 --stats

# 解释两篇文档共享哪些 shingle（含各自出现次数，确定性排序）
python3 -m diskfm.simcli explain my.sim doc_017 doc_023

# 逐文档诊断（短文档被显式标记为不可比较）
python3 -m diskfm.simcli docs my.sim --all

# 深度校验 / 清单与计数器
python3 -m diskfm.simcli validate my.sim
python3 -m diskfm.simcli stats my.sim
```

## Shingle 定义与规则声明

- shingle = 连续 `q` 个 Unicode 码点（`q` 为正整数，构建参数）。
- 原字符串的大小写、空白、换行、NUL 与所有合法码点**原样保留**：
  不分词、不正规化、不折叠大小写（manifest `parameters` 中声明
  `tokenization=unicode-codepoint-qgram/v1`、`normalization=none`、
  `case_folding=none`、`cross_document_shingles=false`）。
- shingle 不跨文档终点。
- 同一文档中的重复 shingle **只计一次**（集合语义）；其出现次数
  （term frequency）另行保存在 posting 中，仅供 `explain` 展示，
  **绝不参与** Jaccard 计数。
- 少于 `q` 个码点的文档没有可比较的 shingle：在 `docs.bin` 的
  `FLAG_NO_SHINGLES`、manifest 的 `documents.without_shingles` 和
  `docs` 诊断输出中显式标记，而不是悄悄当成零相似。

## 相似度与阈值

两篇有 shingle 的文档 A、B 的 Jaccard 相似度 = `|A∩B| / |A∪B|`。
阈值必须落在 `(0, 1]`；浮点输入按其十进制表示解释（`0.1` 就是 1/10，
不是二进制近似），字符串接受 `"0.75"` 或 `"3/4"`。阈值过滤用整数交叉
相乘 `inter*den >= num*union`，结果排序用 `fractions.Fraction` 精确
比较——**二进制浮点和遍历顺序都不会决定边界结果**。结果只包含相似度
不低于阈值的不同 doc_id 对；source 文档永远不会是自己的结果；完全
相同的文档作为不同 doc_id 分别出现。

查询结果按 **相似度降序、再按目标 doc_id 的 Unicode 码点序** 稳定分页，
每行包含：`source_doc_id`、`target_doc_id`、`intersection_count`、
`union_count`、相似度的确定性表示（`similarity="num/den"` 最简分数及
其分子分母）、两边的 `shingle_count`。

## qgram 身份：完整字节表示，无哈希碰撞

本实现选择**路线 A**：qgram 的身份就是其**长度前缀的完整 UTF-8 字节
串**，直接作为外部排序键，并在落盘字典 `qgrams.bin` 中原样保存。
字典按 UTF-8 字节序（即码点序）严格递增。

- 两个不同的 qgram 的字节串必然不同，排序时必然分开——**不存在
  “不同 qgram 被静默合并”的可能**，无需碰撞分桶，也无需复核。
- 打开目录时 `_validate_geometry` 会流式重读整个字典并验证严格递增；
  `validate` 进一步校验 posting/正排关系的有序性与计数一致性。
- 固定 q 下，等长码点序列的 UTF-8 编码互不为前缀（UTF-8 自同步），
  但变长外排框架本身对任意字节键（含互为前缀的键）都正确，并有
  专门测试覆盖。

## 构建路线（全部外存）

1. **输入校验**：以只读方式打开源索引——校验其 OPEN 完成标志、
   manifest、每个文件的长度与 SHA-256、文档表几何；随后流式读取
   文档时**重算源索引的 content_hash 并与封存 manifest 比对**
   （原文摘要校验），同时计算独立的文档集合摘要
   （`documents.digest`）。可选 `--expect-identity` 钉住源索引实例
   身份。源索引不会被修改。
2. **关系生成**：逐文档流式生成 `(qgram_utf8, doc_index)` 原始关系，
   写入有界变长外排器（`chunk_items` 条或 `chunk_bytes` 字节即溢写
   一个有序 run）。文档元信息（id、长度、原始关系数）写入临时文件，
   不在内存累积。
3. **去重 + 字典 + posting**：一趟归并完成：相同 `(qgram, doc)` 合并
   （term_freq 计数）；每个不同 qgram 依序落盘 `qgrams.bin` 与
   `qgramdir.bin`；每个 `(qgram, doc)` 对向 `postings.bin` 追加
   `(doc_index, term_freq)`，并向第二个外排器发射正排关系
   `(doc_index, qgram_id)`。posting 逐条流式写出，单个热门 qgram 的
   超长 posting 不占内存。
4. **正排关系 + 文档统计**：第二趟归并产出每篇文档的有序 qgram id
   列表（`docqgrams.bin`）与 `docs.bin` 定长记录；doc_id blob 第二遍
   流式追加。
5. **原子发布**：manifest（含每个数据文件的 SHA-256 与长度）落盘后，
   最后写入并 fsync `OPEN` 完成标志，再 `os.replace` 暂存目录。
   构建中断、临时空间耗尽、输入损坏或发布前失败都会删除整个暂存
   目录——不会留下可读的部分目录，也不会触碰已存在的有效目录
   （目标已存在时直接 `FileExistsError`）。

## 查询路线（只读目录结构）

给定 `source_doc_id`：

1. 只读取 source 文档自己的正排关系切片（`docqgrams.bin` 中一段）；
2. 只读取这些 qgram 的 posting（分页流式）；候选计数写入**有界外部
   计数归并**（热门 shingle 的长 posting 触发溢写/多路归并），不在
   内存保存任何候选计数哈希表；
3. 归并得到 `(候选doc, 交集)`，并集 = `s + t - 交集`（`docs.bin` 分页
   读取目标 shingle 数）；精确有理阈值过滤；
4. 通过阈值的结果经有界外排按 `(-相似度, doc_id)` 排序后分页。
   **没有候选上限**：共享至少一个 shingle 的每篇文档都会被计数、
   被精确评分，热门模板不会静默漏报。

没有共享 shingle 时（含 source 文档本身无 shingle）立即返回空，
posting 读取计数为 0——不会扫描全体文档拼出假候选。

游标是不透明令牌，绑定目录实例身份 + source_doc_id + 精确阈值
（`num/den`）+ 页内偏移，并带自身 SHA-256 摘要；换目录、换源文档、
换阈值或篡改都会抛 `CursorError`。重算确定性保证跨进程分页稳定。

## 磁盘格式（similarity 版本 1）

每个数据文件以 8 字节头开始：`magic=0x5349('SI'), version=1, kind,
reserved`。详见 `diskfm/simformat.py` 模块文档。

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 格式版本、目录实例 UUID、源索引身份（实例 UUID + content_hash + 几何）、参数（q 与规则声明）、文档集合摘要、关系统计、构建统计、每个文件的 SHA-256+长度 |
| `OPEN` | 完成标志（最后写入并 fsync）；缺失即“构建中断”，目录拒绝打开 |
| `qgrams.bin` | 字典：全部不同 qgram 的原始 UTF-8 字节，按字典序首尾相接 |
| `qgramdir.bin` | 32B 定长记录：blob 偏移、字节长、保留位、posting 偏移与长度 |
| `postings.bin` | 8B 条目 `(u32 doc_index, u32 term_freq)`，按 qgram 分组、组内按 doc_index 有序 |
| `docqgrams.bin` | 每文档的有序 distinct qgram id（u32），按文档序拼接 |
| `docs.bin` | 40B 定长记录（shingle 数、正排偏移、文本码点长、id 偏移、id 长、标志位）+ doc_id blob |

**身份绑定**：manifest 的 `files` 节钉住每个数据文件的长度与
SHA-256；`qgramdir[i]` 同时锚定 `qgrams.bin` 的第 i 段字节与
`postings.bin` 的第 i 段 posting；`docs.bin[d]` 锚定 `docqgrams.bin`
的第 d 段与 id blob 的第 d 段；游标绑定 manifest 中的目录实例 UUID。
打开时即流式校验：每个数据文件的 magic/version/kind 头、文件哈希、
目录几何（posting/正排切片完整平铺、blob 连续）、字典严格递增、
文档标志位与长度一致、id blob 连续；`validate` 再做全量 posting/正排
有序性、doc_id 唯一性与计数一致性校验。任何截断/越界/损坏都抛
`IndexCorruptError`。

**内存/磁盘边界**：构建与查询的排序缓冲区分别受
`chunk_items`/`chunk_bytes` 与 `counter_chunk_items`/
`counter_chunk_bytes`/`result_chunk_items`/`result_chunk_bytes` 限制；
所有随机访问（qgramdir、postings、docs、docqgrams）都经过固定页数的
LRU 分页器，驻留字节数与语料规模无关；posting、字典、正排关系全部
真实落盘并可分页读取。归并扇入受 `merge_fan_in` 限制：run 数超过
扇入时先分组归并成中间 run，文件描述符占用与 run 数无关。峰值
RSS（VmHWM）与溢写字节数由构建统计如实记录。

## 确定性

相同源索引与参数重复构建：所有数据文件字节一致、`documents.digest`
一致、查询结果一致（仅 manifest 中每次构建的目录 UUID 不同）。构建
输出只取决于输入遍历顺序，不依赖任何哈希表顺序；所有排序键都是
全序（qgram 字节 + doc 下标）。

## 统计与诊断

- manifest `build_stats`：文档数（含有无 shingle 分列）、原始/去重
  关系数、字典项、posting 条目、两次外排的 run 数与溢写字节、
  源索引读取字节数、峰值 RSS。
- 查询 `--stats`：source shingle 数、posting 列表/条目读取数、候选
  关系数、计数器溢写 run 数与字节、候选数、精确评分数、结果数、
  结果排序溢写、各分页器命中/未命中/驻留。
- `explain --stats`：共享 shingle 数、字典读取数、posting 探测数。
- `docs`：逐文档 shingle 数、码点长度、`comparable` 标志。

## 测试

```bash
cd tests
python3 -I pytest.py test_similarity.py   # 44 项：语义/分页/游标/损坏/中断/确定性
python3 -I pytest.py test_sim_large.py    # 大语料有界内存证据 + CLI 端到端
```

期望结果由 `tests/sim_oracle.py` 独立生成：对原文逐文档用 Python 集合
求 q 码点片段，独立枚举文档对并用整数交并集计算，完全不接触生产
shingle 生成器、字典、posting 或候选代码。大语料在运行时生成
（热门模板 + 稀疏文档），不提交到 Git。
