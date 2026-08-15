# 回测引擎优化报告

测量日期：2026 年 7 月 22 日

法语原版：[RAPPORT_OPTIMISATION_BACKTEST.md](RAPPORT_OPTIMISATION_BACKTEST.md)

本文件是 `RAPPORT_OPTIMISATION_BACKTEST.md` 的中文版，两者记录相同的实现、测试口径和测量结果。

## 1. 目标

本轮优化在不改变持仓和金融结果的前提下，降低完整回测的运行时间和峰值内存。它延续了第一阶段已经完成的工作：采用轻量结果结构、在每次测试后释放 builder，以及删除临时的 `Unitary_*` score。

同日完成的第二轮优化专门针对包含多个变量和多个期限的批量测试。它将衍生变量集中生成，在不同 signal 之间保留公共月度基础表，并压缩重复标识符的数据类型。

主要优化包含六个方面：

1. 使用 `load_backtest_data(..., start_date, lookback_periods=12)` 只读取需要的时期；
2. builder 以只读方式共享 `screen` 和 `returns`；
3. 每个月只为 Top 和 Worst 准备一次公共数据；
4. 将行业中性排名向量化；
5. 不再把完整矩阵转换成长表，直接计算每日收益；
6. 严格验证 performance、holdings 和 metrics 的等价性。

## 2. 只读取需要的时期

现在推荐这样调用：

```python
screen, returns = load_backtest_data(
    SCREEN_PATH,
    RETURNS_PATH,
    variables=RAW_VARIABLES,
    signal_config=WIDE_CONFIG,
    bench=BENCHMARK,
    start_date=START_DATE,
    lookback_periods=12,
)
```

`start_date` 在读取阶段就过滤收益数据。对于 screen，函数还会保留计算 `pct_N`、`diff_N` 和 `rank_diff_N` 所需要的历史观察期。默认的 `lookback_periods=12` 因此可以覆盖与 1、3、6、12 期之前进行比较的需求。

在实测场景中，列筛选和日期过滤得到：

| 数据 | 优化前 | 优化后 | 减少比例 |
|---|---:|---:|---:|
| screen 行数 | 3,454,342 | 2,255,832 | 34.7% |
| returns 行数 | 5,511 | 4,232 | 23.2% |
| returns 列数 | 1,288 | 1,164 | 9.6% |

只读取技术列、指定变量、可能使用的 denominator，以及 benchmark 成分股所对应的收益列。

## 3. 数据共享与 Top/Worst 公共预处理

builder 初始化时不再深拷贝完整的 `screen` 和 `returns`，而是以只读方式共享这两个对象。只有在确实需要转换数据时，才会创建局部且仅包含必要列的副本。

公共月度预处理主要包括：

- 合并同一家公司的 secondary tickers；
- 过滤 benchmark 成分股；
- 处理市值缺失；
- 平滑权重；
- 计算 benchmark 的行业权重。

这些步骤每个月只执行一次，然后分别供 Top 和 Worst 使用。ESG 和 blacklist 等只属于 Top 的规则仍然只应用于 Top。

## 4. 向量化的行业中性排名

全局排名和行业内排名现在使用 pandas 向量化操作。计算定义没有改变：先计算全局排名并归一化到 0–1，再计算行业内排名并再次归一化到 0–1。

缺失值和相同值的处理方式仍然与参考版本一致。等价性已经在确定性的合成市场和完整真实历史上验证。

## 5. 每日 performance 计算

旧引擎会使用 `stack()` 将 `returns` 和 `returns_drift` 两个完整矩阵转换成长表，然后分别与持仓的每一行合并。这会产生多个非常大的中间表。

新引擎的处理方式是：

1. 通过向量化查找，将每个收益日期对应到最近一次再平衡日期；
2. 以该日期为基准重新基准化累计收益矩阵；
3. 通过行列位置直接读取 drift multiplier 和 return；
4. 避免两次 `stack()` 和对应的两次 merge。

performance 仍然在任何绘图或导出函数之前完成计算。

## 6. 测量方法

优化前后的测试使用完全相同的场景：

- Intel Core i9-13900HX，64 GB 内存；
- 真实数据 `screen_aggregateCIQ.parquet` 和 `returns.parquet`；
- benchmark 为 `STOXX EUROPE 600`，只计算一次后显式传入；
- 2010 年 1 月 1 日至 2026 年 7 月；
- 单变量测试 `Revenue 5Y CAGR | level`；
- Top 和 Worst 比例均为 13%；
- 断点年份为 2020、2022、2024；
- `fill_method="copy"`，关闭图片，不保留 builders。

每 10 毫秒采样一次进程 RSS。“新增峰值”是该步骤的峰值 RSS 减去步骤开始时的 RSS。优化前的数据来自本阶段修改之前保存的 reference snapshot。

## 7. 测量结果

| 步骤 | 优化前时间 | 优化后时间 | 时间收益 | 优化前新增峰值 | 优化后新增峰值 | 峰值收益 |
|---|---:|---:|---:|---:|---:|---:|
| 数据读取 | 0.888 秒 | 0.791 秒 | 10.8% | 690.1 MiB | 616.8 MiB | 10.6% |
| Benchmark | 10.187 秒 | 5.530 秒 | 45.7% | 1,544.6 MiB | 1,011.6 MiB | 34.5% |
| 完整单变量测试 | 62.021 秒 | 36.647 秒 | **40.9%** | 1,691.2 MiB | 616.0 MiB | **63.6%** |

三个步骤的累计时间从 **73.10 秒降到 42.97 秒**，减少 **41.2%**。观察到的最高绝对 RSS 从 **2,458.7 MiB 降到 1,655.5 MiB**，减少 **32.7%**。

单变量测试完成后仍然保留的内存从 60.1 MiB 降到 42.4 MiB，进一步减少 29.5%。第一阶段已经避免在每个结果中长期保留完整 builder，这里的收益是在其基础上继续获得的。

单次测量会受到机器负载和磁盘缓存影响。最稳定的结构性收益来自删除长表中间数据，以及减少读取的行数和列数。

## 8. 结果严格等价

完整真实历史使用 `pandas.testing.assert_frame_equal(..., check_exact=True)` 比较优化前后的对象，所有检查均完全一致：

| 检查对象 | 结果 |
|---|---|
| Top、Worst 和 Benchmark performance | 完全一致 |
| Top 历史持仓 | 完全一致 |
| Worst 历史持仓 | 完全一致 |
| Top/Benchmark、Worst/Benchmark、Top/Worst ratios | 完全一致 |
| 经典 metrics | 完全一致 |
| 分时期 metrics | 完全一致 |
| Robust score 及相关 metrics | 完全一致 |

主要数值保持不变：

| 指标 | 优化前 | 优化后 |
|---|---:|---:|
| Robust score | -0.7371 | -0.7371 |
| Top/Benchmark | -0.0665 | -0.0665 |
| Top/Worst | 0.0845 | 0.0845 |

自动化测试还为 performance、Top/Worst holdings、经典 metrics 和时期结果固定了可重复的 SHA-256 指纹。覆盖范围包括显式传入的 benchmark、输入共享、公共月度预处理、ISIN 作为 parquet index 的情况、变量方向、Composition 和时期重构。

## 9. 历史行业中性算法审计

历史算法在原有定义范围内并不是错误的，但需要明确它实际保证的内容。它首先将 score 转换为可比较的行业内排名，然后在全局选择最高或最低排名，最后将组合行业权重调整为 benchmark 行业权重。因此，它主要保证选择后的**权重中性**，并不保证每个行业恰好选择 13% 的股票。

对 `STOXX EUROPE 600` 中 `Revenue 5Y CAGR` 的 209 个月、3,971 个行业月份分组进行审计后发现：

- benchmark 中不存在缺少 ICB 19 行业的记录；
- 572 个分组至少存在一组相同值；
- 19 个分组没有任何有效原始值，全部出现在 2026 年 4 月；
- 在最后一个月，目标比例为 13%，但各行业实际选择比例介于 11.1% 和 16.7%；
- 在原始历史中，如果整个横截面都不可用，某个月可能选中 score 缺失的股票。

主要风险包括：

1. 当一个行业少于两个不同 score 时，min-max 归一化会返回 `NaN`；
2. `nlargest()` 或 `nsmallest()` 可能根据原始行顺序，用缺失值补足目标数量；
3. 阈值处相同值没有显式的业务级次级排序规则；
4. 目标数量在部分资格过滤之前计算，可能改变最终有效 percentile；
5. 当前中性化使用行业，但没有使用真正多区域 benchmark 所需要的“区域 × 行业”组合；
6. 如果之后还会重新计算行业内排名，那么最初的全局排名及其归一化是冗余的。

为了保证历史回测等价，本次向量化有意保留了原有定义。当行业缺失时，也和旧循环一样保留全局排名。

未来可以通过独立选项引入一套新定义，避免破坏历史结果。建议包括：

1. 先执行资格过滤，再计算配额；
2. 按日期和分组检查覆盖率、有效值数量及不同值数量；
3. 为无效分组定义明确策略：报错、沿用上期组合或使用中性 score；
4. 直接在“区域 × 行业”分组内排序，不再先做全局排序；
5. 使用最大余额法分配分组配额，同时满足行业 percentile 和总持仓数量；
6. 使用稳定且有文档说明的次级键处理相同值，例如先按市值、再按 ISIN；
7. 将覆盖率、配额、相同值和 fallback 诊断一起导出。

## 10. 衍生变量 profiling

独立 profiling 使用 10 个真实变量、13 个维度 `level`、`pct_N`、`diff_N`、`rank_diff_N`，以及 1、3、6、12 四个期限。受控窗口从 2025 年 7 月 1 日开始，包含 130,072 行，最终在 `screen` 中保留 120 个衍生列。

优化前，每个衍生维度都会重新创建并排序自己的表。一项变量如果需要 12 个衍生维度，就会触发 12 次排序。新函数只准备一次原始值和 rank，只按 ISIN、Date 排序一次，然后从同一有序表生成所有期限。

| Profiling 指标 | 优化前 | 优化后 | 收益 |
|---|---:|---:|---:|
| 总时间 | 41.400 秒 | 16.445 秒 | **60.3%** |
| 排序次数 | 120 | 10 | **91.7%** |
| 新增峰值 | 383.1 MiB | 383.1 MiB | 不变 |
| 保留内存 | 139.1 MiB | 134.5 MiB | 3.3% |

峰值保持不变，是因为 120 个最终衍生列本来就需要保留在 `screen` 中。主要收益来自减少计算和临时表。输出的 130 个列——10 个原始 level 加 120 个衍生列——均通过 `check_exact=True` 与 reference 严格比较，结果完全一致。

## 11. Signal 之间的公共月度基础表

`monthly_base_cache` 按日期和回测 universe 保存已经准备好的技术列：日期、行业、benchmark 权重和市值。当前 signal 的 score 只在运行该测试时附加到基础表上。因此，Top、Worst 和后续 signals 都可以复用相同的月度准备。

缓存与 `screen` 对象绑定。如果传入另一个 screen，缓存会自动清空。显式设置为 `None` 可以关闭缓存。在 Notebook 中，将同一个字典放入 `RUN_OPTIONS`，即可在 unitary、incremental、composite 调用之间共享。

缓存的中间版本曾让每个月的表都保留完整的 ISIN 全局分类，因此占用 361.1 MiB。最终版本使用局部普通 ISIN index，并删除不参与月度选股的 SEDOL。198 个月的缓存现在只占 **9.5 MiB**，相对于该中间版本减少 **97.4%**。

在完整真实测试中，第一个 signal 用 11.52 秒建立缓存。相同 universe 和设置下，第二个 signal 复用缓存后需要 10.92 秒，减少 5.2%。整体收益有限是合理的：经过前面几轮优化后，每个 signal 的主要耗时已经转移到每日 performance 和 metrics 计算。

## 12. 标识符和行业的紧凑数据类型

`load_backtest_data(..., compact_dtypes=True)` 默认开启，使用：

- `CategoricalIndex` 保存 ISIN；
- `category` 保存 SEDOL 和区域；
- 在行业代码为整数且范围允许时，使用 nullable `Int8` 保存 supersector 和 industry；
- 金融变量和 returns 保留原始数值类型，避免损失精度。

对于 2,255,832 行、7 列的筛选后 screen，pandas 测得的内存从 485.7 MiB 降到 **72.8 MiB**，减少 **85.0%**。数据读取步骤实际保留的 RSS 从上一轮测量的 555.4 MiB 降到 356.8 MiB。分类转换可能略微增加一次性读取时间，但会让后续整个批量测试受益。

## 13. 新一轮完整测量和等价性

完整测量继续使用第 6 节定义的场景。下表比较的是已经优化过的上一版本和当前版本：

| 步骤 | 上一轮版本 | 当前版本 | 变化 |
|---|---:|---:|---:|
| 数据读取 | 0.791 秒 | 1.338 秒 | 包含紧凑类型转换 |
| 数据读取新增峰值 | 616.8 MiB | 535.4 MiB | **-13.2%** |
| Benchmark | 5.530 秒 | 3.140 秒 | **-43.2%** |
| Benchmark 新增峰值 | 1,011.6 MiB | 811.8 MiB | **-19.8%** |
| 第一个完整 signal | 36.647 秒 | 11.520 秒 | **-68.6%** |
| 第一个 signal 新增峰值 | 616.0 MiB | 487.9 MiB | **-20.8%** |
| 后续 signal 使用缓存 | — | 10.918 秒 | 复用公共月度基础表 |

单次耗时仍会受到磁盘缓存和机器负载影响。最可靠的结构性指标是排序次数、screen 的 pandas 大小，以及月度缓存大小。

修改后，完整真实历史的 performance、Top/Worst holdings、ratios、经典 metrics、分时期 metrics 和 robust score 的所有标量仍然严格一致。自动化测试目前包含 27 个测试和 2 个参数保护子测试，明确覆盖一次排序、紧凑类型、月度基础表复用和并行回测。

## 14. 修改前后的代码对比

下面的代码片段只保留产生主要成本的语句。`...` 仅表示省略未变化或与比较无关的代码。“修改前”对应引擎优化之前的 snapshot，“修改后”对应 commit `f3447f0`。

### 14.1 按列、日期读取并使用紧凑类型

修改前已经筛选需要的列，但会读取 screen 的全部历史，而且重复标识符仍然是 `object`：

```python
screen = pd.read_parquet(
    screen_path,
    columns=requested_columns,
)
returns = pd.read_parquet(
    returns_path,
    columns=return_columns,
)
```

修改后，parquet filter 只保留需要的时期和 lookback，然后压缩标识符和行业类型：

```python
screen_start_date = resolved_start_date - pd.DateOffset(
    months=int(lookback_periods),
)
screen_filters = [('Date', '>=', screen_start_date.to_pydatetime())]

screen = pd.read_parquet(
    screen_path,
    columns=requested_columns,
    filters=screen_filters,
)
if compact_dtypes:
    screen = compact_screen_dtypes(screen)

returns = pd.read_parquet(
    returns_path,
    columns=return_columns,
)
returns = returns.loc[returns.index >= resolved_start_date]
```

实测效果：screen 行数减少 34.7%，筛选后 screen 的 pandas 大小从 485.7 MiB 降到 72.8 MiB。

### 14.2 共享输入和可复用月度准备

修改前，每个 builder 都会完整复制两个大表，每个月的调用还会再次复制当月 screen：

```python
self.screen = copy.deepcopy(screen)
self.returns = copy.deepcopy(returns)

screen = copy.deepcopy(screen_agg_monthly)
```

修改后，builders 以只读方式共享输入。月度技术基础表只计算一次，读取缓存时只附加当前 signal 的 score：

```python
self.screen = screen
self.returns = returns
self.monthly_base_cache = monthly_base_cache

cache_key = self._monthly_base_cache_key(raw_date)
if cache_key is not None and cache_key in self.monthly_base_cache:
    preparation = self._preparation_from_monthly_base(
        self.monthly_base_cache[cache_key],
        source_screen,
        list_score_col,
    )
    return self._finalize_sec_list_spot(preparation)
```

实测效果：198 个月的缓存只占 9.5 MiB。第一个 signal 包含缓存建立需要 11.52 秒，后续同 universe signal 需要 10.92 秒。

### 14.3 每个变量只排序一次，同时生成全部期限

修改前，按 ISIN 和 Date 排序位于维度循环内部。同一变量的 12 个衍生维度会触发 12 次排序：

```python
for component, column in components:
    if component != 'level':
        ordered = pd.DataFrame({
            '_position': np.arange(len(screen)),
            '_isin': isin_values,
            '_date': pd.to_datetime(screen['Date']).to_numpy(),
            '_value': source_values.to_numpy(),
        }).sort_values(['_isin', '_date'])

        if base_dimension == 'pct':
            filled_values = ordered.groupby('_isin')['_value'].ffill()
            derived = filled_values.groupby(ordered['_isin']).pct_change(
                periods=period,
                fill_method=None,
            )
        else:
            derived = ordered.groupby('_isin')['_value'].diff(period)
        screen[column] = pd.Series(
            derived.to_numpy(),
            index=ordered['_position'],
        ).sort_index().to_numpy()
```

修改后，先排序再进入维度循环。group、forward fill 和 rank 均只准备一次，然后一起生成所有期限：

```python
ordered = pd.DataFrame(ordered_data).sort_values(['_isin', '_date'])
ordered_groups = ordered.groupby('_isin')
filled_values = (
    ordered_groups['_value'].ffill()
    if any(dimension.startswith('pct_') for dimension in dimensions)
    else None
)

derivatives = {}
for dimension in dimensions:
    base_dimension, period_text = dimension.rsplit('_', 1)
    period = int(period_text)
    if base_dimension == 'pct':
        derived = filled_values.groupby(ordered['_isin']).pct_change(
            periods=period,
            fill_method=None,
        )
    elif base_dimension == 'rank_diff':
        derived = ordered_groups['_rank_value'].diff(periods=period)
    else:
        derived = ordered_groups['_value'].diff(periods=period)
    derivatives[f'{variable}__{dimension}'] = pd.Series(
        derived.to_numpy(),
        index=ordered['_position'],
    ).sort_index().to_numpy()

screen[list(derivatives)] = pd.DataFrame(derivatives, index=screen.index)
```

实测效果：profiling 场景中的排序次数从 120 次降到 10 次，衍生变量生成时间从 41.40 秒降到 16.45 秒。

### 14.4 不使用长表计算每日收益

修改前，两个完整矩阵会转换成长表，然后与持仓合并：

```python
returns_drift_flat = (
    returns_drift.stack().to_frame().reset_index()
)
returns_flat = (
    df_returns.stack().to_frame().reset_index()
)
df_merge = df_merge.merge(
    returns_drift_flat,
    how='left',
    on=[col_date, col_id],
)
df_merge = df_merge.merge(
    returns_flat,
    how='left',
    on=[col_date, col_id],
)
```

修改后，日期和股票代码先转换为位置，然后直接读取两个 NumPy 矩阵：

```python
date_positions = returns_drift.index.get_indexer(
    pd.DatetimeIndex(df_merge[col_date]),
)
security_positions = returns_drift.columns.get_indexer(df_merge[col_id])
valid_positions = (date_positions >= 0) & (security_positions >= 0)

drift_values = np.full(len(df_merge), np.nan)
return_values = np.full(len(df_merge), np.nan)
drift_matrix = returns_drift.to_numpy(copy=False)
return_matrix = df_returns.to_numpy(copy=False)
drift_values[valid_positions] = drift_matrix[
    date_positions[valid_positions],
    security_positions[valid_positions],
]
return_values[valid_positions] = return_matrix[
    date_positions[valid_positions],
    security_positions[valid_positions],
]
```

与第一轮其他优化共同测量时，完整单变量测试从 62.02 秒降到 36.65 秒，新增峰值从 1,691.2 MiB 降到 616.0 MiB。第二轮优化又将其降到 11.52 秒和 487.9 MiB。

## 15. 修改前后的时间汇总

下表汇总了在相同真实历史和相同 `Revenue 5Y CAGR | level` 测试下测得的三个版本。“初始版本”位于引擎向量化之前；“第一轮优化”对应第 2–5 节；“当前版本”进一步加入一次排序、紧凑类型和月度缓存。

| 步骤 | 初始版本 | 第一轮优化 | 当前版本 | 初始 → 当前收益 |
|---|---:|---:|---:|---:|
| 数据读取 | 0.888 秒 | 0.791 秒 | 1.338 秒 | -50.7% |
| Benchmark | 10.187 秒 | 5.530 秒 | 3.140 秒 | **69.2%** |
| 第一个完整 signal | 62.021 秒 | 36.647 秒 | 11.520 秒 | **81.4%** |
| **实测总时间** | **73.096 秒** | **42.968 秒** | **15.998 秒** | **78.1%** |

当前读取比初始版本慢 0.45 秒，因为其中包含紧凑类型转换。这是一次性成本，却可以避免整个批量测试期间额外保留超过 400 MiB，因此不能单独理解为整体性能退化。

按时间顺序理解：

1. 第一轮将总时间从 73.10 秒降到 42.97 秒，减少 41.2%；
2. 第二轮进一步降到 16.00 秒，相比第一轮减少 62.8%；
3. 从初始版本到当前版本累计减少 78.1%，整体约快 4.6 倍。

内存变化方向相同：

| 内存指标 | 初始版本 | 第一轮优化 | 当前版本 | 初始 → 当前收益 |
|---|---:|---:|---:|---:|
| 数据读取新增峰值 | 690.1 MiB | 616.8 MiB | 535.4 MiB | 22.4% |
| Benchmark 新增峰值 | 1,544.6 MiB | 1,011.6 MiB | 811.8 MiB | 47.4% |
| 第一个 signal 新增峰值 | 1,691.2 MiB | 616.0 MiB | 487.9 MiB | **71.2%** |
| 最高绝对 RSS | 2,458.7 MiB | 1,655.5 MiB | 1,261.8 MiB | **48.7%** |

第 10 节的 10 个变量、13 个维度 profiling 使用另一个包含 130,072 行的窗口。它的优化前 41.40 秒和优化后 16.45 秒不能与上面的 15.998 秒相加，因为这是一项专门隔离衍生列生成成本的测量。

## 16. 不同 signal 的并行回测

`test_unitary_signals()`、`test_incremental_signals()` 和 `test_composite_signals()` 现在都支持 `n_jobs`。默认 `n_jobs=1` 严格保留原来的串行行为；大于 1 时，将相互独立的 signal 回测分配给多个进程。

并行层级只放在真正独立的任务上：

- 同一变量的衍生维度仍然只排序一次并一起生成；
- Top 和 Worst 仍然属于同一任务，共享月度准备；
- 父进程只计算一次 benchmark；
- 缓存为空时，第一个 signal 负责建立月度缓存；
- workers 接收紧凑 screen、returns、benchmark 和一份可直接读取的缓存副本；
- 无论进程实际完成顺序如何，结果都按原始配置顺序合并；
- Plotly 图形返回主进程后，也按该顺序显示。

核心变化可以概括为：

修改前：

```python
results = {}
for variable in signal_config:
    results[variable] = run_top_worst_backtest(...)
```

修改后：

```python
with ProcessPoolExecutor(
    max_workers=worker_count,
    initializer=_init_worker_context,
    initargs=(screen, returns, list_noire_path, execution_options),
) as executor:
    ordered_results = list(
        executor.map(_run_worker_signal, signal_tasks)
    )
```

第一个真实验证场景在 2010–2026 完整历史上运行 4 个 `level` signal：`Revenue 5Y CAGR`、`FCF Conversion`、`Gross Margin` 和 `Ebitda Margin`。两次执行使用相同 benchmark、断点和全部设置。

| 4 个 signal 批量测试 | 时间 | 加速倍数 | 时间收益 |
|---|---:|---:|---:|
| `n_jobs=1` | 39.039 秒 | 1.00× | — |
| `n_jobs=4` | 22.332 秒 | **1.75×** | **42.8%** |

第一个 signal 保持串行，用于预热月度缓存；其余三个同时执行。并行时间包含 Windows 进程创建和向每个 worker 序列化紧凑数据的成本。

第二个 benchmark 更接近日常 unitary 批量测试：10 个变量、默认 4 个维度 `level`、`pct_1`、`diff_1`、`rank_diff_1`，共 40 个关闭图片的完整回测。

| 40 个 unitary 测试 | 时间 | 8 相对 6 的加速 | 8 相对 6 的时间收益 |
|---|---:|---:|---:|
| `n_jobs=6` | 215.509 秒 | 1.00× | — |
| `n_jobs=8` | 132.912 秒 | **1.62×** | **38.3%** |

40 个结果仍然严格一致。对于这台测量机器上的大批量 unitary 测试，推荐使用 `n_jobs=8`。如果任务数量较少，使用 2–4 个进程可以避免创建超过实际可并行任务数量的 workers。

严格等价检查覆盖 performance、Top/Worst holdings、ratios、经典 metrics、分时期 metrics 和 Composition。自动化测试分别验证 unitary、incremental、composite 批次、结果顺序和 Plotly figure 返回。并行流程也已经在 Windows 的真实 Jupyter kernel 中执行验证。

即使输入已经压缩，每个进程仍然拥有自己的工作表。`n_jobs > 1` 时禁止使用 `retain_builders=True` 和单一 `save_path`；应当先合并全部结果，再统一导出。

## 17. 限制与后续优化方向

剩余成本主要集中在每个 signal 重复进行每日 performance 计算，以及历史代码中仍然存在的部分 `groupby.apply()` 和日期循环。

后续优化应继续使用相同的严格等价协议：

1. 将剩余的 selection 和 weighting `groupby.apply()` 替换为更直接的分组操作；
2. 在数学定义允许时，共享多个组合之间相同的每日计算步骤；
3. 增加非阻塞的自动 benchmark，用于监控时间和资源回退，同时避免让功能测试变得不稳定；
4. 只有当跨多次 kernel 重启的批量测试确实需要时，再考虑可选的持久化缓存。
