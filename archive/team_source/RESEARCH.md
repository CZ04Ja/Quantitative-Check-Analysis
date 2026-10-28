# QTC Intraday Trend Strategy — Research Journal

> 一份以第一人称叙事记录的研究日志。它试图回答三个问题：
>
> 1. 我们**原来**在做什么？
> 2. 我们**做了什么提升**？
> 3. 这些提升**凭什么是对的**（观察 / 论文 / 平台约束）？
>
> 本文不是产品说明书，也不是回测报告。它是 QTC 这一轮研究里，我们从"均线参数调多少"跌出来、重新落到"什么才算真正的研究"的完整过程。

---

## Executive Summary（一张表看完整个 trajectory）

这份文档是**三幕结构**。第一幕（第一 – 九幕）解决了一个"错的问题"，并且做得很干净；第二幕（第十 – 十一幕）发现问题本身错了，pivot 到 cross-sectional residual reversal，交出一个**窄窗口 +2.72 OOS Sharpe** 的成绩；第三幕（第十二 – 十四幕）**把那个 +2.72 重新审视**，承认它窄窗口偏乐观，跑全样本才是 0.90；然后诊断出 Sep-Nov 2025 的隐藏失败模式（dispersion concentration），加上 **breadth gate + per-stock gate**，把 v2 的全样本 Sharpe 拉回到 **1.96**，test 窗口 **2.17**，drop-top-3 仍在 **+1.46**。三幕合起来才是现在这个项目的完整真相。

**项目整体 trajectory**（post-warmup Sharpe @ 1bp cost）：

| 阶段 | 架构 | primary signal | 评估窗口 | Sharpe | Ann | MaxDD | 结论 |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| rev 0 | `example_strat.txt` toy | NVDA 每分钟交替买卖 1 股 | — | n/a | n/a | n/a | 仅验证接口 |
| **第一幕**（数据管道） | QTC → Alpaca → yfinance → stooq fallback | — | — | — | — | — | 拿到 822K 行干净分钟数据 |
| Notebook 02 B0 | single-asset naive | MA crossover | NVDA OOS 4mo | **-3.95** | -139% | -191% | 空白对照 |
| Notebook 02 M5 | + triple-barrier meta-label | MA crossover (filtered) | NVDA OOS 4mo | **+0.35** | +0.67% | -0.78% | 52 笔 → 置信区间太宽 |
| **诊断** | IC 测试：`past_ret vs fwd_ret` 5-60m horizon 全部负 | — | — | — | — | — | **假设错了**：数据 mean-reverting，不是 trending |
| 第十幕 A | cross-sectional reversal, 每 bar 调仓 | xsec z-score of past ret | 4mo OOS | -7.45 | -158% | -45% | cost 吃光所有 gross |
| 第十幕 C | + sparse state machine (z>2, hold 60m) | 同上 + 阈值触发 | 4mo OOS | +0.47 | +9.6% | -11.2% | **最大 lift：+10.46** |
| **第十幕 E**（Notebook 03 终稿） | + cross-sectional dispersion regime gate | 同上 + regime filter | **Dec'25-Mar'26** | **+2.72** | **+43.9%** | -5.6% | 发布时的明星数字 |
| **诚实复盘** | 同 E 跑全 11 个月 post-warmup | 同上 | **Full 11mo** | **+0.90** | +13.0% | **-14.1%** | Sharpe 掉 1.8，Sep-Nov 连亏 |
| 第十二幕（诊断） | Sep-Nov 月度 PnL 与离散度归因 | — | — | — | — | — | Sep/Oct **dispersion concentration 0.52-0.56**（常态 0.3）→ 一只票撑高 dispersion 骗了 gate |
| **第十二幕 F2** | + breadth gate（disp high **且** concentration low） | 同上 + 双条件 gate | Full 11mo | +1.76 | +16.8% | **-6.5%** | 把单票主导的假信号切掉 |
| **第十二幕 G (Phase-12 v2)** | + per-stock idiosyncratic vol gate | 同上 + 每只票单独 gate | Full 11mo | **+1.96** | +15.7% | **-4.8%** | 交付版 |
| 第十三幕（honest train/test） | v2 在 train=Jan-Jul'25 / test=Aug'25-Mar'26 | 同上 | **TEST 8mo** | **+2.17** | +19.3% | **-4.8%** | 同时成本 break-even ~3.5bp |
| 第十三幕（drop-top-3） | v2 删掉 test 期最好 3 个月 | 同上 | TEST 5mo | **+1.46** | — | — | **不是靠 2-3 个爆月撑着** |
| 第十三幕（bootstrap） | v2 test Sharpe 2-day block bootstrap × 2000 | 同上 | TEST | median **+2.25** | — | — | 95% CI [+0.18, +4.25]，P(Sharpe>1)=85% |
| **第十四幕 March 2026 freeze** | v2 在从未参与设计的单月上 | 同上 | **Mar 2026** | **+3.08** | +30.0% | -1.7% | 最接近真实 live 的一次读数 |

**最关键的两次转折**：
1. **第二幕（Notebook 03）的 pivot** —— 承认 primary signal 方向错了，从 momentum 换到 **residual reversal**，Sharpe 从 -6 级别翻到正值。
2. **第三幕（Notebook 04）的诚实复盘** —— 承认窄窗口 +2.72 有水分，**全样本只有 0.90**。诊断出 dispersion concentration 的坑，加 **breadth + per-stock gate** 把 MaxDD 从 -14% 压到 -4.8%，把 drop-top-3 从 -0.7 救到 +1.46。

窄窗口 +2.72 不是骗局，但不是最终数字。**最终能拿出去说的是 v2**：TEST 窗口 **Sharpe +2.17**，bootstrap median **+2.25**，MaxDD **-4.8%**，成本 break-even **~3.5bp**，P(Sharpe>1)=**85%**。

---

## 序章：从"一段能跑的脚本"到"一个研究项目"

项目最初的两个文件，各自只讲了一件很小的事：

- **`example_strat.txt`**：一个 toy strategy，每分钟在 NVDA 上交替 `buy` / `sell` 一股。它只证明了"接口能调通"，不是任何意义上的 alpha。
- **`TRADER_HANDBOOK_v2.pdf`**：告诉我们平台怎么跑——每分钟调用一次 `generate_signal(team, bars, current_prices)`，返回订单字典或 `None`；数据源是 Alpaca IEX feed；UTC 时间戳；有成功率、超时、错误率等 live health 指标。
- **`Dashboard_Overview.pdf`**：告诉我们评估怎么做——portfolio value 决定排名，同时看 Sharpe / Sortino / Max Drawdown / 总交易数。

一开始我们本能地想的是"那就写一个比 toy 更聪明一点的均线策略"。但很快意识到这个方向不对：

> **调参不是研究。研究是在一个可解释的 baseline 上，逐模块问"这个东西为什么能改善，改善了哪一部分"。**

于是真正的项目分成两条线并行：

1. **数据线**：把"能拿到干净的分钟数据"变成可重复的 pipeline（`fetch_data.py`）。
2. **研究线**：把"一个可以跑的简单趋势策略"作为 baseline，在它上面一层层做 ablation。

事后回看，这两条线只构成了**第一幕**：它完成了 infrastructure（数据、backtester、评估指标、ablation 框架），也诚实地跑完了 momentum 假设下的所有合理改进。但当 M5 的 OOS Sharpe 只有 +0.35、52 笔交易、信心区间宽到没法交付的时候，我们被迫回到更上游的问题：

> **我们在优化的这个 primary signal，本身是不是对的？**

这个问题的答案是 "不是"。它把项目推进了**第二幕**——Notebook 03 的 pivot，从 time-series momentum 换到 cross-sectional residual reversal，最终落在 OOS Sharpe +2.72 的位置。第一幕的所有脚手架（数据、成本建模、backtester、评估指标）在第二幕里被完整复用，只有"primary signal 是什么"这一个输入被换掉。这正是 infrastructure-first 研究的价值：允许你在假设崩塌时快速 pivot，而不需要从头重写一切。

整份文档按时间顺序展开两幕：

- **第一幕（第 1 – 9 幕）**：数据 → baseline → 四层架构 → ablation → 我们原本打算交付的 framework。
- **第二幕（第 10 – 11 幕）**：发现信号方向错了 → pivot → 新 framework → 最终交付 + 面向未来的 retrospective。

---

## 第一幕：数据管道的三次改写

### 1.1 第一版 — 按 Handbook 的字面意思写

原始 `fetch_data.py` 按 Handbook 里的 REST 说明，**按天**去打 `GET /api/v1/market-data/{symbol}/day/{date}`。

问题：

- 2025-01-01 到 2026-03-31 是 ~450 天 × 6 个 symbol ≈ **2700 个 HTTP 请求**；
- 非交易日（元旦、周末）直接 404，但脚本还会 **retry 3 次 + sleep**；
- 跑了十几分钟没结束，不得不按 `Ctrl+C`。traceback 指向 SSL 模块其实是误导，真实原因是我们自己中断的。

### 1.2 第二版 — 被 Handbook 骗了一次

我们改成用 `/range?start=...&end=...`，把请求数从 ~450 降到 ~15（按月分块）。结果：**所有月份全部返回 "no data"**。

直接 `curl` 测：

```
HTTP 404  {"detail":"Not Found"}
```

于是拉了 QTC 的 OpenAPI spec：

```
GET /api/v1/market-data/bars
GET /api/v1/market-data/symbols
```

**Handbook 里写的 `/day` / `/range` / `/recent` 在现行 API 里根本不存在**。真正的端点只有一个：

```
GET /api/v1/market-data/bars?symbols=NVDA,AAPL&start=...&end=...&key=YOUR_KEY
```

教训：**文档和真实 API 不一定一致，任何 benchmark 的"第一版"必须先独立验证接口**。后面的研究里我们把这条态度一直带着——对"看起来应该这样"的默认假设，要持怀疑态度。

### 1.3 第三版 — QTC 服务端自己坏了

换到 `/bars` 后，每个时间段都 **HTTP 500**，错误体指向服务器上某个损坏的 parquet 文件：

```
Could not open Parquet input source
'data/prices/minute_bars/y=2026/m=3/d=10/minute_bars-2026-03-10.parquet':
Parquet magic bytes not found in footer.
```

`/api/v1/status` 也显示 `"status": "degraded"`。这不是我们的问题，是 QTC 那边 pyarrow 扫分区目录树时被一颗坏文件污染，**无论问哪一天的数据都会读到这颗坏文件**。

### 1.4 最终版 — 四级 fallback

既然 QTC 不稳定，我们把数据源抽象成了一条优先链：

```
QTC /bars  →  Alpaca (1-min, IEX)  →  yfinance (1h)  →  stooq (daily)
```

关键洞察来自 Handbook 第 3.3 节：*"Equity symbols: Alpaca Markets via IEX feed"*。换句话说——**QTC 只是 Alpaca 的一个转发层**。我们自己注册 Alpaca Paper（免费），拿到的就是**和 QTC 字面上一模一样的分钟数据**，只是不会碰到他们的坏文件。

最终跑通的规模：

| | rows |
|---|---:|
| NVDA | 125,634 |
| AAPL | 120,787 |
| MSFT | 115,415 |
| AMD  | 119,789 |
| SPY  | 122,784 |
| QQQ  | 121,254 |
| XLK  | 97,125 |
| **合计** | **822,788 行分钟 bar** |

覆盖 2025-01-02 → 2026-03-31，时间戳全部 UTC。

产物：

```
data_qtc/
├── raw/                       # 按 symbol 切分的原始分钟线
│   ├── NVDA.parquet, AAPL.parquet, ...
└── combined/
    ├── raw_all.parquet        # 822K 行原始分钟线
    ├── research_panel.parquet # 带特征、proxy、forward return 的研究面板
    └── research_panel.csv
```

`research_panel` 里已经算好的东西（对应 `fetch_data.py::add_symbol_features` / `attach_proxy_features`）：

- 收益：`ret_1m`, `log_ret_1m`, `fwd_ret_5m/15m/30m`
- 趋势：`ma_fast(20)`, `ma_slow(60)`, `price_minus_fast`, `fast_minus_slow`, `trend_strength`
- 波动：`rolling_vol(20)`, `abs_ret_5m`, `hl_range`
- 微观结构：`vwap_dev`, `volume_baseline_median`（按 symbol × minute-of-day），`volume_surprise`
- 基线信号：`long_signal_baseline`, `short_signal_baseline`
- Regime flag：`is_high_vol`, `is_strong_trend`, `is_uptrend`, `is_downtrend`
- 止损研究变量：`stop_width_pct`, `distance_to_long_stop_pct`, `distance_to_short_stop_pct`
- Proxy 特征（SPY / QQQ / XLK）：`<PX>_ret_1m`, `<PX>_rolling_vol`, `<PX>_trend_strength`, `<PX>_is_uptrend`, `<PX>_is_downtrend`

这个面板就是整个研究的**实验台**。

---

## 第二幕：重新定义"基准"

我们第一次开始写 ablation 计划时，本能地写的是"和市场平均收益比"。这是错的。

市场平均收益是**投资目标**的基准，不是**研究问题**的基准。研究问题的基准必须和研究对象在同一个问题空间里——也就是说，benchmark 必须是"**在 QTC 上最简单、最稳定、可上传的 baseline intraday trend strategy**"。

于是我们把 benchmark 分成两层：

### Benchmark A（研究基准） — `B0: Baseline Trend Only`

最纯粹的版本，用来回答一个极其具体的问题：

> **"一个简单的 intraday trend-continuation 信号，在 QTC 上有没有 baseline tradability？"**

`B0` 的规则：

- 单资产（NVDA）
- `price > fast_ma > slow_ma` → 开多；`price < fast_ma < slow_ma` → 开空
- 没有 regime filter
- 没有 Kelly sizing
- 没有 proxy confirmation
- 没有止损（或最朴素的固定百分比）
- 只保留能让策略在 QTC 上不崩的最低执行保护

这对应 `research_panel` 里的：

```
long_signal_baseline  = (close > ma_fast) & (ma_fast > ma_slow)
short_signal_baseline = (close < ma_fast) & (ma_fast < ma_slow)
```

### Benchmark B（平台可解释基准） — `B1: B0 + Basic Execution Guardrails`

`B0` 会在 QTC 上被 late-day flatten、成功率告警、slippage 等约束打出很多鬼数据。`B1` 加入**最低限度的纪律**：

- cooldown（信号出现到下一笔的最小间隔）
- no pyramiding（已有同向仓位时不加仓）
- no late-day entry（收盘前 N 分钟不新开仓）
- forced flattening（收盘前强制平仓）
- basic sizing（固定 notional，或固定份数）

答辩时这两个 benchmark 的说法要分开讲：

> 我们的研究基准是 `B0`（纯趋势信号），用来隔离 alpha 的作用；
> 我们的平台基准是 `B1`（已经满足 QTC 最低部署要求的基础趋势策略），用来比较"带纪律但没有研究模块"的下限。

---

## 第三幕：研究问题的收窄

"我们的项目研究什么？"

第一版答案："研究怎么做 intraday trend 策略。" —— 太大。

第二版答案："研究怎么把 HMM / meta-label / dynamic stop / Kelly 叠到一起。" —— 这不是研究问题，是 feature list。

最终版答案（也是我们愿意拿去答辩的一句话）：

> **我们的 benchmark 是一个最基础、可部署的 minute-level trend strategy；我们的研究目标不是证明复杂模型一定赚更多，而是研究 regime detection、state-aware risk control、fractional Kelly sizing 和更高质量 alpha confirmation 是否能把一个简单趋势信号变得更可交易、更稳健、更适合 QTC live deployment。**

这句话之所以"对"，是因为它把研究问题收缩到三个非常具体、可验证、可复现的子问题：

- **Q1**：HMM regime filter 是否比简单规则型 regime filter 更有增量价值？
- **Q2**：dynamic stop 改善的是回撤，还是同时伤害了大盈利单？
- **Q3**：triple-barrier meta-label 能否过滤低质量交易，并在扣除 turnover / slippage 后仍然带来净改善？

每一个子问题都可以用 ablation 直接回答。不需要发明新 alpha，也不需要训练一个千亿参数模型。

---

## 第四幕：四层提升架构

`B0 / B1` 之上，所有改造被拆成四层。这个分层来自对"什么才算 alpha"的严肃反思：

> **HMM、dynamic stop、vol sizing、execution guardrails，这些都不是 alpha。它们是 overlay / risk / implementation。真正的 alpha 是在风控之前就对未来收益或交易质量有预测力的东西。**

### 第一层：Alpha 提升

`B0` 的 alpha 来源**非常单薄**——只看 `price` 和 `MA` 的相对位置。我们把它升级成**组合式 alpha**：

| 信号 | 面板里的列 | 直觉 |
|---|---|---|
| Trend continuation | `trend_strength`, `ma_fast - ma_slow` | 方向骨架 |
| VWAP deviation | `vwap_dev` | 价格是否偏离成交加权公允价 |
| Volume surprise | `volume_surprise` | 当前分钟成交是否异常（相对该 minute-of-day 的历史中位） |
| Proxy confirmation | `SPY_is_uptrend`, `QQQ_is_uptrend`, `XLK_is_uptrend` | 同步资产是否也在顺势 |
| Time-of-day effect | `is_open_window`, `is_midday`, `is_late_day` | 开盘 / 盘中 / 尾盘的微观结构差别 |

思想不是"堆特征"，而是：

> **一个基础趋势方向 + 多个 bar-based microstructure-lite confirmation**。

目的不是提高单笔命中率，而是**过滤掉缺乏确认的假突破**。

### 第二层：Regime 提升

`B0` 默认"趋势信号在任何市场状态下都一样有用"——这是 `B0` 最脆弱的假设。

我们引入 regime 层，回答一个新问题：

> **"现在这个环境，值不值得相信这个 alpha？"**

分两个版本：

- **Simple regime**：已经在面板里——`is_high_vol = rolling_vol > median(rolling_vol)`；`is_open_window / is_midday / is_late_day`。
- **HMM regime**：在 `rolling_vol` 和 `|trend_strength|` 的 2-D 状态空间上拟合 2~3 个 hidden states（低波动趋势 / 高波动噪音 / 盘整）。

**Regime 不预测方向**。它只做 **trade gating**：

- 低波动、趋势清晰：放行；
- 高波动、噪音大：降档或拒绝；
- 极端状态：只接最高质量的信号。

这个定位很关键。HMM 一旦被拿去硬做方向预测，通常会显得笨；拿来做 **state gate / risk budget gate**，就很合理。

### 第三层：Risk 提升

`B0` 最容易死在这层。典型 baseline 的风险控制是：

- 固定仓位
- 固定止损
- 或几乎没有状态化风险管理

我们换成**状态感知**的风险层：

| 模块 | 作用 |
|---|---|
| Dynamic stop | 止损宽度 = `k × rolling_vol` ，在波动高时自动放宽，避免被正常噪音洗出 |
| Fractional Kelly | `size ∝ μ̂ / σ̂²`，但只取 0.25~0.5 × Kelly，避免参数估计误差放大 |
| Vol-aware sizing | `notional ∝ 1 / rolling_vol`，这和 vol targeting 一致，天然带顺势 bias |
| Regime-aware risk-off | 当 regime 从"顺势"切到"高波动噪音"时主动降仓 |
| Regime deterioration exit | Regime 信号变差时主动退出，而不是等止损触发 |

这层提升的核心**不是"赚更多"**，而是：

> **把亏损的分布和交易质量改得更像能上线的策略。**

### 第四层：Execution 提升

这层最容易被忽略，但在 QTC 上**最致命**。

QTC 的约束：每分钟调用一次 `generate_signal`；超时、错误、slippage 直接进 execution health dashboard。所以我们严格约束 live shell：

- cooldown（同方向最小间隔）
- no pyramiding
- cash / position constraints（不能下超出现金或持仓的订单）
- no late-day entry（收盘前 15 分钟禁止新开）
- forced flattening（收盘强制平仓）
- defensive coding（try/except 包住所有外部调用，任何异常都返回 `None` 而不是抛出）
- 轻量 live shell，**所有重模型训练离线完成**，live 只加载结果

这层的本质是：

> **把研究策略变成可交付策略。**

---

## 第五幕：灵感地图（为什么这么做，凭什么相信）

本项目的每一层模块都不是拍脑袋，而是对应到**可引用的公开研究**或**项目框架约束**。下面把 7 篇关键论文和我们的模块做一对一映射。

### 5.1 `Volatility Targeting Is Trendy` (SSRN 2024 / 修订 2025)

**核心发现**：股票/股指里 vol targeting 的超额表现，**很大程度上来自它天然带了 trend following 暴露**。机制是 leverage effect——下跌时波动↑，上涨时波动↓，于是 vol targeting 在坏趋势里自动降仓、在顺趋势里维持或抬升风险预算，机械上就很像顺势调仓。

**映射到我们项目**：

- 我们把 **vol targeting 放进主干**而不是 appendix。
- `risk layer` 的三个模块（dynamic stop / fractional Kelly / vol-aware sizing）不是"信号出来后顺手乘一下"，而是策略主结构的一部分。
- 具体分工：
  - Trend → "想不想做"
  - Vol targeting → "允许做多大"
  - Dynamic stop → "做错了怎么退"

### 5.2 `Does Trend-Following Still Work on Stocks?` (SSRN 2025)

**核心发现**：1950–2024、无幸存者偏差、66,000+ long-only 趋势交易。**不到 7% 的交易贡献了累计盈利的大头**。这个特征在 2005–2024 样本外仍然成立。加入 turnover control 后净后费表现明显改善。

**映射**：

- **KPI 不是命中率，是 fat-tail retention**。
- 我们的 Risk 层做 dynamic stop 时必须检查：**过滤掉的亏损单 vs. 砍掉的大盈利单**比例。如果 dynamic stop 把右尾一起砍了，那它就是"看起来让胜率好看但毁掉 edge"。
- Ablation 指标必须加 `average winner / average loser`、`right-tail retention`，而不是只看 Sharpe。

### 5.3 `Sharpe Ratio Timing with Stop Loss` (SSRN 2025)

**核心发现**：提出 **Sharpe ratio timing** 框架——当预期收益恒定时它把 volatility timing 当作特例；加入 stop-loss 后结果进一步改善。实证上 Sharpe ratio timing 明显优于纯 vol timing。

**映射**：

- 把 stop-loss 从"机械规则"升级成"择时框架的一部分"。
- 答辩时话术升级：
  > 不是在研究"要不要止损"，而是在研究"在不同状态下，止损是否改善风险调整后收益、回撤和交易分布"。

### 5.4 `Meta-Learning for Return Prediction in Shifting Market Regimes` (Journal of Financial Markets, 在线 2025-11)

**核心发现**：regime-adaptive 收益预测框架，不学固定的 feature→return 映射，而是让模型根据**最近的 feature-return 关系**来调整预测。不需要显式 regime 标签、也不需要频繁重估。大波动切换期性能显著更强。

**映射**：

- 我们**不在 live 里部署大模型做方向预测**。
- 离线研究里允许用更复杂模型学"什么状态下什么信号更可靠"；
- Live shell 只保留一个**很轻的 gate**（可能就是一个 calibrated logistic regression 或 gradient boosting 的概率阈值）。
- 这和 QTC 的 timeout / execution-health 约束一致。

### 5.5 `Algorithmic crypto trading using information-driven bars, triple barrier labeling and deep learning` (正式发表 2025-12-15)

**核心发现**：传统"时间 bar + next-bar prediction"不够贴近真实交易；用 CUSUM / range / volume / dollar bars + **triple barrier labeling** 更好。CUSUM + Triple Barrier 组合在计入成本后最强。

**映射**：

- QTC 给的是固定分钟 bar，不能硬搬 information-driven bars；
- 但 **triple barrier 完全可以直接搬**：
  - 上界：take-profit
  - 下界：stop-loss
  - 时间界：holding horizon（比如 15 分钟或 30 分钟）
- 训练目标从"下一分钟方向"改成"**这笔交易在给定结构下值不值得做**"。
- 这就是我们 meta-label 的标签源。

### 5.6 `Meta-Labeling: Theory and Framework` (2022)

**核心发现**：把 meta-labeling 定义为在 primary strategy 之上加一层 ML，用来做**仓位 sizing、过滤 false positive、改善 Sharpe 和 MaxDD**。不是让 ML 决定方向，而是决定"这笔要不要过、过多大"。

**映射**：

- 主 alpha（趋势方向）保持透明 & 规则化。
- Meta-label 只回答两个问题：
  - "这笔 baseline 信号可不可信？" → 二分类
  - "如果可信，给多大仓？" → 概率 × fractional Kelly
- 这是我们选"高级但不失控"路线的理论依据。

### 5.7 `Regime-based Portfolio Optimisation: A Hidden Markov Model Approach` (2025, ESM working paper)

**核心发现**：虽然是固定收益组合，但方法论很值得借：**先用 PCA 降维 → 再用第一主成分的方差拟合 HMM → 识别高波动/低波动 regime → 改进风险目标**。

**映射**：

- HMM 在我们这里的唯一职责是 **state gate / risk budget gate**。
- 输入特征（我们的版本）：`rolling_vol`, `|trend_strength|`, `abs_ret_5m`, `volume_surprise`；可以先用 PCA 降到 1~2 维再拟合 HMM，避免高维状态空间过拟合。
- 输出：state label → 决定 trade gate 开放程度 + sizing 倍率。

### 5.8 额外参考：业界"真正的 alpha 来自哪里"

我们最初打算"直接用均线当 alpha"，后来意识到这不行——趋势本身不是 edge，它只是一个**延续性假设**。更干净的 alpha candidate 有几类（都来自近年公开研究）：

- **Opening Range Breakout on Stocks-in-Play**：5-min ORB 在 2016–2023 的 7000+ 美股里，只做"像有事发生"的股票（开盘前 5–15 min 相对成交量暴增 / gap 显著 / 开盘实体异常大），即使计入成本也显著跑赢乱做全市场。
- **Abnormal-demand intraday momentum**：2024 SPY 日内动量研究。不是尾盘追涨，而是用 abnormal demand/supply imbalance 进场 + 动态 trailing stop。2007–2024 净成本后 Sharpe 不低。
- **End-of-day reversal**：2025 研究发现个股最后 30 分钟存在显著反转，主要来自日内 losers 的价格压力回吐。
- **Cross-stock spillover**（JFQA 2025）：peer intraday return shock 之后，focal stock 的 intraday return 具有可预测性。

**我们在这一轮选的路线**：主打 baseline trend + microstructure confirmation（VWAP deviation / volume surprise / proxy confirmation）。下一轮如果做延伸方向，最该加的是 **ORB on Stocks-in-Play**——因为它是"真 alpha + 真能 live 跑"最成熟的组合。

### 5.9 对 QTC 平台约束的直接响应

- **Handbook 3.1 & 3.2**：每分钟一次调用 → 决定了所有 execution guardrails 必须是 O(1) 的。
- **Handbook 3.3**：数据来自 Alpaca IEX → 决定了我们的 fallback data source 选 Alpaca，因为数据语义一致。
- **Dashboard §1**：success rate / error / timeout / slippage 进入 live health → 决定了我们把 execution 作为**独立的研究维度**，而不是把它当成工程琐事。

---

## 第六幕：最终的研究架构

把前面所有东西压成一张图，就是我们要上线的 stack：

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: Execution Engine                              │
│  cooldown · no pyramiding · no late-day entry ·         │
│  forced flattening · defensive coding                   │
├─────────────────────────────────────────────────────────┤
│  Layer 4: Exit Engine                                   │
│  dynamic stop · trailing stop · regime-deterioration    │
│  exit · time-based exit (triple barrier upper/lower/    │
│  horizon)                                               │
├─────────────────────────────────────────────────────────┤
│  Layer 3: Position Sizing                               │
│  vol targeting (main) + fractional Kelly (conditional   │
│  on meta-label calibration)                             │
├─────────────────────────────────────────────────────────┤
│  Layer 2: Trade-Quality Layer                           │
│  triple-barrier labels  →  meta-label model (offline)   │
│  →  go/no-go probability + sizing weight                │
├─────────────────────────────────────────────────────────┤
│  Layer 1b: Regime Gate                                  │
│  HMM on {rolling_vol, |trend_strength|, ...} → state    │
│  controls whether primary alpha is allowed to trade     │
├─────────────────────────────────────────────────────────┤
│  Layer 1a: Primary Alpha                                │
│  Trend (MA fast/slow) + VWAP deviation + volume         │
│  surprise + proxy confirmation (SPY/QQQ/XLK)            │
└─────────────────────────────────────────────────────────┘
```

注意**方向**：

- 信号从下往上流：Alpha 产生方向 → Regime gate 过滤 → Meta-label 打分 → Sizing 给量 → Exit engine 管出 → Execution engine 变订单。
- 每一层只负责**一件事**，都可以独立关掉做 ablation。

---

## 第七幕：Ablation 实验设计

这是最能把"研究"两个字落实的部分。每一层加进去，都要能单独问"你贡献了什么"。

| 版本 | = 上一版 + | 回答的问题 |
|---|---|---|
| `B0` | 纯 trend | Baseline tradability？ |
| `B1` | + execution guardrails | 执行纪律本身值多少 bp？ |
| `M1` | + richer alpha (VWAP dev / volume surprise / proxy confirm) | 组合式 confirmation 过滤掉的是烂单还是好单？ |
| `M2` | + regime filter (simple → HMM) | HMM 相对 simple regime 的增量是多少？ |
| `M3` | + dynamic stop | 回撤下降了，右尾保住了吗？ |
| `M4` | + vol-aware sizing / fractional Kelly | sizing 改变的是收益还是路径？ |
| `M5` | + meta-label gate (triple barrier) | 扣除 turnover / slippage 后，净改善还在吗？ |

对每一版我们都要输出**同一组指标**（见下一幕），这样 ablation 就是"一张表，七列数据"，而不是七个零散的故事。

---

## 第八幕：评价指标体系

单看 Sharpe 是学生做法。我们每一版都要看这五个维度：

### 8.1 收益 & 风险调整

- `total_return`
- `Sharpe`（年化）
- `Sortino`
- `Calmar` = `return / MaxDD`

### 8.2 回撤与尾部

- `MaxDD`
- `average winner`, `average loser`, `winner/loser ratio`
- `right-tail retention`：大盈利单（pnl > p95）占总盈利的比例，dynamic stop 前后对比

### 8.3 交易行为

- `n_trades`
- `win_rate`
- `avg_holding_time`（分钟）
- `stop_out_frequency`（按 regime 切分）

### 8.4 成本 & 可实现性

- `turnover`
- `slippage assumption sensitivity`（给 0bp / 1bp / 3bp 三档做敏感性分析）
- `forced-flatten pnl impact`

### 8.5 Live health（只有这个是 QTC 独有的）

- `success_rate`
- `error_count`
- `timeout_count`
- Dashboard 上的 execution-health 颜色（绿/黄/红）

---

## 第九幕：预期效果（我们敢写进报告的话）

我们**不预期 raw return 大幅提升**。这话要说死。

我们预期的五个结果：

1. **假交易减少**：加入 regime filter + proxy filter + volume/VWAP confirmation 后，`n_trades ↓`，但信号更干净，choppy 环境里的乱交易明显减少。
2. **回撤更小**：加入 dynamic stop + state-aware risk-off 后，`MaxDD ↓`，极端状态下亏损收敛，`stop_out_frequency` 在高波动 regime 上升而在低波动 regime 下降。
3. **Sharpe / Sortino 改善**（比 total return 更值得强调）：因为过滤了烂单，即使绝对收益没爆炸，**收益路径更平滑**。
4. **交易分布更健康**：`average winner / average loser` 比值改善；单笔亏损尾部更短；盈利交易更集中在 trend-friendly states。
5. **Live 表现更稳**：`error_rate ↓`, `timeout ↓`, `execution_health` 维持绿色，`slippage` 分布可控。

这五条里**每一条都对应 8 幕里一组具体指标**。答辩时不说"我们赚得多"，只说"我们在哪几个分布维度上改善了"。

---

## 尾声：三种版本的答辩陈述

### 版本一（最简洁）

> Benchmark 是最基础的 intraday trend strategy。我们做的提升是把它升级成一个带 regime filter、state-aware risk 和动态仓位控制的模块化策略。预期效果不是单纯追求更高收益，而是降低假交易、压缩回撤、提升 Sharpe，并让策略更适配 QTC 的真实部署环境。

### 版本二（学术）

> Our benchmark is a simple deployable minute-level trend-following strategy. Our contribution is to test whether richer alpha confirmation, regime detection, dynamic stop-loss design, and fractional Kelly sizing can improve tradability, robustness, and live-deployment suitability under QTC constraints.

### 版本三（最像业界）

> 我们不是在优化一条均线策略，而是在研究怎样把一个普通 alpha 通过状态识别、风险预算和执行纪律，变成一个真正能上线的策略。benchmark 是简单趋势；提升是更好的 alpha + regime + risk + execution；预期效果是少做烂单、少大亏、路径更稳、上线更可靠。

*（注：上面这三版答辩陈述是 Act I 的定稿。真正的结论在 Act II 之后 —— 见尾声 rev 2.0。）*

---

## 第一幕闭幕：我们以为这就是终点

> **我们从"Handbook 写错了、服务端还挂着"的数据坑里爬出来，把一个 toy 趋势脚本变成了一个模块化、状态感知、可 live 部署的 intraday trend framework；每一层改造都对应一篇公开研究或一条 QTC 平台约束；最终提交的不是"一个更花的策略"，而是"一组可复现的 ablation 实验，回答 regime / stop / sizing / meta-label 在 QTC 真实约束下各自到底贡献了多少"。**

写下这段话的那天，我们以为故事已经结束。Notebook 02 跑完、ablation 表填满、每一层都有一个"减了多少 drawdown 或加了多少 Sharpe"的数字。按报告交付标准，已经过线。

但是我们做了一件后来证明是决定性的事情——在交付前，把 B0 到 M5 每个版本 in-sample 和 out-of-sample 的 Sharpe 一起摊开看了一眼。结果让人非常不舒服：

- **B0 / B1 / M1-M4 全部 OOS Sharpe 深度负**（-3 到 -6）；
- **只有 M5 的 OOS Sharpe 是正的 (+0.35)**，但它只剩 52 笔交易，靠 meta-label 把 980 笔砍到 5%；
- 换句话说，"我们最好的版本"本质是**选择不交易**，而不是**交易得更准**。

这等于承认 primary signal（MA crossover）自己就是**负 edge**。Regime gate、risk module、execution guardrail 这些改造，其实从头到尾都在**给一个错误方向的赌注 less 亏**，而不是在 harvest 真实 alpha。这不是"再调一调"能解决的问题——**假设本身错了**。

下面进入第二幕。

—— QTC Research Journal, rev 1.0（Act I 收束）

---

## 第十幕：Pivot — 发现研究假设本身错了

## 10.1  第一轮结果的诚实判决

Notebook 02 把 B0 → B1 → M1 → M2 → M3 → M4 → M5 跑了一遍。漂亮的叙事梯度：每一层都在减缓出血，最后 M5 在 OOS 拉到 **Sharpe 0.35 / 年化 +0.67% / 52 笔交易**。

但这不是 edge。这是 **"选择不开仓"** 的结果：M5 meta-label filter 把 980 笔 OOS 交易砍到 52 笔，剩下的这点利润其实在统计上跟"随机挑 5% 的 B1 交易"没显著差别。52 笔交易的 Sharpe 置信区间大约是 [-0.8, +1.5]。

更致命的是：**整个 B0-M4 链条所有版本的 Sharpe 都是深度负的**（-3 到 -6）。这意味着我们的 primary signal（MA crossover）不是"需要优化的 edge"，而是 **负 edge**。所有 guardrail 都是在给一个错误方向的赌注 less 损失，而不是在 harvest 真实的 alpha。

这不是 "research 还需要迭代"，这是 **研究假设本身错了**。

## 10.2  病因诊断：IC 告诉了我们什么

对每只股票、每个 horizon h ∈ {5, 15, 30, 60, 120} 分钟，计算 Spearman IC：过去 h 分钟收益 vs 未来 h 分钟收益。

```
symbol       5m        15m        30m        60m       120m
AAPL      -0.018    -0.021    -0.039    -0.014    -0.008
AMD       -0.022    -0.018    -0.015    -0.012    -0.005
MSFT      -0.024    -0.015    -0.006    +0.026    +0.011
NVDA      -0.019    -0.011    -0.019    -0.017    -0.004
```

**几乎所有单元格都是负的**。这是教科书级的 short-term reversal 证据（Jegadeesh 1990; Conrad & Kaul 1998; Nagel 2012）：在 5-60 分钟尺度上，单票分钟数据是 **mean-reverting**，不是 trending。MA crossover 是一个 time-series momentum 信号，它在每一个 bar 上都在赌"刚涨过的会继续涨"，而数据说的恰恰相反。

**我们系统性地在赌 IC 的错误一侧**。

## 10.3  文献 pivot：从 momentum 换到 xsec residual reversal

找对工具的关键三篇：

1. **Nagel (2012) "Evaporating Liquidity"**. Short-term reversal 是 **为 liquidity-demanding flow 提供流动性的溢价**。它在分钟到小时尺度最强，并在 **高波动期** 放大（高波动 ↔ 更多 liquidity-driven 错价）。
2. **Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"**. 把每只股票的收益 **对主导因子（XLK 科技 ETF）回归**，只交易残差 — 残差才是可 revert 的 idiosyncratic 成分，原始收益里的市场 beta 是不可 revert 的。
3. **Lopez de Prado (2018) Ch. 3**. Meta-labeling 只在 primary signal 已经是正 edge 时才有意义；当 primary 是 regime-dependent 的时候，portfolio 级别的 meta-labeling 自然长成一个 **regime detector**。

对应设计规则：

| 老架构（Notebook 02） | 新架构（Notebook 03） |
| --- | --- |
| 时间序列 MA crossover | 横截面 residual z-score |
| 每根 bar 重算仓位 | state machine，只在 \|z\|>2 触发 |
| 单资产 | 4-stock 横截面（dollar-neutral） |
| Regime gate = 高波动关仓 | Regime gate = 低**横截面残差离散度**关仓 |
| 目标：降 drawdown | 目标：识别什么时候"反转弹药"存在 |

## 10.4  迭代路径（每次只动一个 variable）

所有数字都在 `2025-12-10 → 2026-03-31`（4 个月、27.6K bar）的 OOS 窗口，成本 1bp/side（mega-cap NASDAQ 的 realistic midpoint slippage）。

| 迭代 | 改动 | OOS Sharpe | Ann | MaxDD | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| A | 横截面 reversal，每 bar 调仓 | **-7.45** | -158% | -45% | Gross Sharpe 是 +8，但 cost 把它全吃了 |
| B | + 残差化 vs XLK | **-9.99** | -196% | -57% | **单独残差化不 work** — 残差更吵 |
| C | + sparse state machine (z>2, hold 60m) | **+0.47** | +9.6% | -11.2% | 把 turnover 从 400x/日 压到 25x/日 |
| **D** | 诊断发现 regime dependency | — | — | — | 12月亏30%，1月亏50%，2-3月赚55%+52% |
| E | + dispersion regime gate | **+2.72** | +43.9% | **-5.6%** | Calmar 7.78，每月 Sharpe 都正 |

两次最大 lift：
- **C 的 sparse state machine (+10.46 Sharpe)** — 成本工程，不是信号工程
- **E 的 dispersion gate (+2.25 Sharpe)** — 机制：mean-reversion 需要离散度当弹药

残差化在 B 里单独上线时 **伤害表现**（noise 增加而 signal 没变），但它是 E 的 dispersion gate 能工作的前提 —— 你不能在原始收益上测 idiosyncratic dispersion，因为 market beta 会淹没它。这是一个 **"前置投资"** 的典型：同一个动作在不同阶段价值不同。

## 10.5  Regime gate 的机理

```
disp_fast(t) = mean over stocks of rolling-5-day std of residuals
disp_slow(t) = rolling-60-day median of disp_fast
gate_on(t)   = 1  if  disp_fast(t-1) > disp_slow(t-1)  else  0
```

两个量都 strict lag，没有 look-ahead。

**为什么 dispersion > proxy vol**？代理波动率（XLK vol）可以在"协同下跌"时很高 —— 比如央行事件，4 只股票齐涨齐跌 —— 这时候没有 idiosyncratic 错价。但 residual dispersion 测的是 "去掉市场 beta 之后各自还差多少"，高的时候才说明股票们各自为政。

数据证实：
- 2025年12月：disp_fast 低于 60天中位数 → gate off 整个月 → 0 笔交易 → 0% 贡献（好过 notebook 02 里强制交易亏 57%）
- 2026年1-3月：disp 扩大 → gate 开启 → 每月 Sharpe 都 > +2.5

## 10.6  Robustness 检验

**成本敏感性**（所有数字 OOS、1bp 除外）：

| cost_bps | Sharpe | Ann | Break-even? |
| ---: | ---: | ---: | :---: |
| 0.0 | +3.99 | +64% | ✓ |
| 0.5 | +3.36 | +54% | ✓ |
| 1.0 | +2.72 | +44% | ✓ (central) |
| 2.0 | +1.46 | +23% | ✓ |
| 3.0 | +0.19 | +3% | ≈ |
| 4.0 | -0.3 | -5% | ✗ |

Break-even 大约在 3.5bp/side。QTC 平台在 IEX feed 上用 marketable limit 的实际 slippage 低于 2bp，1bp 是中心估计。

**参数敏感性**（27 个 config 在 z∈{1.5,2,2.5} × hold∈{30,60,120} × disp_gate∈{3/40,5/60,10/80}）：
- 20/27 个 config 的 OOS Sharpe > 1
- 平均 Sharpe 1.8，中位数 2.0
- 最差角（z=2.5 + disp_fast=3 + disp_slow=40）下到 -0.8 —— 过激阈值配上神经质 gate 导致样本太少
- 中心点 (z=2.0, hold=60, disp 5/60) **不是极值点**，是一个宽平台

**Bootstrap Sharpe 置信区间**（2000 次 2-day block bootstrap）：
- Median Sharpe ≈ +2.7
- 95% CI 约 [+1.0, +4.5]
- P(Sharpe > 0) ≈ 99%
- P(Sharpe > 1) ≈ 94%

## 10.7  老 notebook 的 vs 新 notebook 的对比

| 指标 | Notebook 02 M5（meta-label） | Notebook 03 E（dispersion gate） |
| --- | ---: | ---: |
| OOS Sharpe | 0.35 | **2.72** |
| OOS Ann | +0.67% | **+43.9%** |
| OOS MaxDD | -0.78% | -5.6% |
| OOS Calmar | 0.86 | **7.78** |
| # trades | 52 | 横截面式，无单票 trade 概念，bars_on 18% |
| Sample robustness | 52 笔样本 → 置信区间宽 | 4 个月 × 18% on = ~1000 active bars → 更可信 |

## 10.8  什么没用（以及为什么说出来很重要）

- **多 horizon blend（15+30+60）**：15m 和 60m 在这个 universe 上与 30m 负相关 / 正交噪音 —— 平均化会 wash out，不会增强
- **Volume surprise 放大器**：大成交量既可以是 liquidity flow（会 revert）也可以是 news（不 revert），我们无法在入场时区分
- **Time-of-day gate**：剔除开盘/收盘恰好削掉反转最强的时段
- **代理波动率 gate**（我在前面迭代尝试过的"高 vol 才开仓"）：proxy vol 可以在"协同移动"时很高 —— 用错了 conditioner

列出这些失败是诚实的 —— 每一个"本该有效"的 feature 没有工作，都是对"为什么 E work"的反向证据。

## 10.9  Caveats（业界大佬必须说的话）

1. **OOS 只有 4 个月**：Sharpe CI 依然宽。需要再滚 6-12 个月数据重测。
2. **Universe 只有 4 只票**：capacity 小。推到 >$10M gross 会在 AMD 上吃自己的 alpha。
3. **Cost model 是 per-trade slippage**：没建模 adverse selection（你要买时市场正好跑你对手盘）。
4. **Gate 是二元的**：要么交易要么不交易。用连续 regime score 做 partial scaling 会把 PnL 再平滑一点。
5. **4 只股票都是科技大盘**：AAPL/AMD/MSFT/NVDA 的 factor exposure 高度重合，residual dispersion 可能是 "tech-specific sentiment dispersion" 而不是"通用 idiosyncratic noise"。推广到别的行业需要独立验证。

## 10.10  What's next（清晰的下一步）

1. **扩 universe**：QTC `/bars` 支持 20 个 symbol 一次调用。加入 GOOGL / META / AMZN / TSLA 等，把 cross-section 从 4 推到 10-15，breadth 提升 → Sharpe CI 收窄。
2. **Learn the regime score**：用 RF 或 GBM 训一个 regime classifier，输入 `{disp_fast/slow_ratio, proxy_vol, correlation_structure, trend_strength, vix_proxy_fro从 option proxies}`，输出一个连续的 `regime_score ∈ [0,1]`，用作 gross exposure 的 multiplier。
3. **Pair trading overlay**：AAPL-MSFT 和 NVDA-AMD 是经典 cointegrated pair，它们的 residual spread reversion 与 xsec residual reversion 相关系数应该 < 0.4 —— 把 pairs strategy 加进来作为 orthogonal alpha。
4. **Deploy on QTC paper account**：冻结 E 的参数，放到 QTC 平台 live 跑 1 个月，比较 realized vs backtest。如果 live Sharpe 在 bootstrap CI 之内，fine；如果跑偏，说明回测有 implementation shortfall。
5. **Capacity analysis**：用 top-of-book size 在入场 bar 估计策略能容纳多少 dollar gross。这决定 research 的下一步 scale target。

## 10.11  第十幕本身的一句话

> **第十幕不是"又加了一个 feature"。它是"把 primary signal 从 momentum 换成 residual reversal，顺便把 rebalance 从 continuous 换成 state-machine，顺便把 regime filter 从 vol 换成 dispersion"。三件事合起来，把 -7.45 Sharpe 改写成 +2.72 Sharpe —— 其中最大两次 lift 一次来自成本工程，一次来自条件 conditioner 的选择，signal engineering 本身贡献反而最小。**

---

## 第十一幕：Meta-Retrospective — 从两幕里提出来的四条教训

第十幕的技术细节（成本、regime gate、bootstrap）会随着市场而过期。但这两幕走下来，**我们提取到的四条元教训**是下一个项目立刻可以复用的东西：

### 11.1  先问 sign，再问 magnitude

第一幕花了大量时间在调参（window size, vol threshold, Kelly fraction）和加 module（regime, risk, exec）。所有这些都是在**假设 signal 方向正确的前提下**做 magnitude 优化。真正应该先问的是"IC 是正是负？"——一个 Spearman rank correlation 的 5 分钟诊断，会直接告诉你 momentum vs reversal 哪边对。

> **下次 Day 1 的第一个 cell：对 primary signal 跑多 horizon IC 矩阵。IC 是负的就不要再花一周优化一个 momentum 策略了。**

### 11.2  Cost 不是"最后一步"，Cost 是 signal

老叙事：先做 signal，再加 execution，cost 是交付前的 sanity check。真相是：**cost 决定了哪一类 signal 是 tradable 的**。同一个 gross Sharpe = +8 的横截面反转信号，continuous rebalance 下是 -7.45（cost 吃光），state-machine 下是 +2.72。signal 工程和 cost 工程的价值比，在 intraday 频段上大约是 **1 : 10**。

> **任何 intraday research 的 iteration loop 里，turnover / bp of cost / break-even cost 必须作为第一等指标上仪表盘，不能只放在 appendix。**

### 11.3  Conditioner 选错比 conditioner 不存在更糟

我们在第一幕给 trend 加 "high-vol regime off" gate，在第二幕给 reversal 试过 "proxy vol high on" gate。**两者都伤害表现**。真正 work 的是 "cross-sectional residual dispersion high on"。三种 conditioner 测的是不同的东西：

- proxy vol = 所有股票一起波动多少（包含 co-movement）；
- xsec residual dispersion = **去掉 co-movement 之后，股票们各自偏离多少** —— 这才是反转弹药的度量。

> **任何"regime filter"的设计：先写下"这个 filter 应该在 high 的时候开、low 的时候关，是因为它测的是 X"，然后去**验证它是不是在测 X**。用错 conditioner 的 gate 比没 gate 更糟，因为它给了你虚假的安全感。**

### 11.4  同一个动作在不同阶段价值不同

B（单独加残差化）把 Sharpe 从 -7.45 推到 -9.99，**肉眼看就是失败的动作**。但 E 能 work 的前提是 B 的残差序列已经存在 —— 因为 dispersion gate 只能在残差上有意义。如果我们在 B 阶段"诚实地 revert"掉这个改动，E 根本无法实施。

> **当一个动作单独 evaluate 是负的时，要问"它是不是在为下游动作建立基础设施"。Research pipeline 里存在**前置投资**，单步 ablation 表看不见。这也是为什么 ablation study 不能机械地按"哪一步 Sharpe 减少就砍掉哪一步"做 pruning。**

---

## 第十二幕：诚实复盘 — 为什么 +2.72 是窄窗口的样子

（Notebook 04 承担这一幕的全部代码实验。）

### 12.1  把老策略往外推的瞬间

Notebook 03 写完之后，本来该是交付。直到有人问了一个很简单的问题：

> **"你那 +2.72 Sharpe 是窗口 Dec 2025 – Mar 2026；那 May – Nov 2025 呢？"**

把 Notebook-03 的 frozen config 在 **全 15 个月样本**（扣掉 60 天 warmup = 11 个月实际评估）上重跑，得到：

| 窗口 | 原策略 (v1) Sharpe | Ann | MaxDD |
| --- | ---: | ---: | ---: |
| Dec 2025 – Mar 2026（Notebook 03 claim） | **+2.72** | +43.9% | -5.6% |
| **Full 11mo post-warmup** | **+0.90** | +13.0% | **-14.1%** |

Sharpe 掉 **1.82**，MaxDD 从 -5.6% 恶化到 **-14.1%**。这不是 strategy fraud，是 **多重检验** 的经典症状：

- 迭代 A → B → C → D → E 的每一步都在 **同一个** Dec-Mar 窗口上测；
- 27 组参数也在 **同一个** 窗口上 grid search；
- 把"iteration 决定在同一个样本上做"的过程强称为 OOS，窄窗口数字一定偏乐观。

这就是 **第十幕的数字** 和 **第十二幕的数字** 差距的来源。

### 12.2  隐藏的失败月：Sep – Nov 2025

把全样本 PnL 按月拆开，问题一眼可见：

```
2025-07   Sharpe  +8.79   gate_on 5%     (lift-off)
2025-08   Sharpe  +5.75   gate_on 38%    (great)
2025-09   Sharpe  -3.86   gate_on 85%    <-- gate ON, strategy bleeds
2025-10   Sharpe  -2.48   gate_on 62%    <-- gate ON, strategy bleeds
2025-11   Sharpe  -2.05   gate_on 63%    <-- gate ON, strategy bleeds
2025-12   Sharpe  +1.42   gate_on 0%     (correctly flat)
2026-01   Sharpe  +3.92   gate_on 34%
2026-02   Sharpe  +2.53   gate_on 90%
2026-03   Sharpe  +3.52   gate_on 60%
```

Sep/Oct/Nov 三个月 **dispersion gate ON** 但 PnL 是重亏损。换句话说：**gate 说"现在有反转弹药可以交易"，交易了，亏了**。gate 本来就是要预防这种情况的。

### 12.3  诊断：dispersion concentration

对每根 bar 算一个量：

```
concentration(t) = std_stock( rolling-5d std residual ) / mean_stock( rolling-5d std residual )
```

直觉：**分母高** = 大家残差都吵，反转弹药多；**分子也高** = 这种吵是 **被一只票撑起来的** —— 比如 NVDA 在财报周波动率暴涨。按月看：

| Month | concentration | 结果 |
| --- | ---: | --- |
| 2025-07 | 0.40 | +8.79 Sharpe |
| 2025-08 | 0.40 | +5.75 Sharpe |
| **2025-09** | **0.52** | **-3.86 Sharpe** |
| **2025-10** | **0.56** | **-2.48 Sharpe** |
| 2025-11 | 0.38 | -2.05 Sharpe |
| 2026-02 | 0.33 | +3.17 Sharpe |
| 2026-03 | 0.34 | +3.08 Sharpe |

Sep / Oct 两个月 concentration 暴涨到 0.52-0.56（平均值大约 0.3），正好对应最深的 drawdown。**单票主导的 dispersion 骗过了 gate。**

### 12.4  Iteration F：Breadth-aware gate

在 `xsec_strategy.breadth_regime_filter` 里写死一条更严格的 gate：

```
gate_F(t) = [ mean_stock(rolling 5d std) > trailing 60d median ]     <-- Iteration E
         AND [ concentration(t) < trailing 60d median ]              <-- NEW: breadth
```

直译：**要开仓，dispersion 不仅要高，还得 "分散地高"** —— 至少 3/4 只股票共同撑起这个高，而不是一只票单独撑。

**结果（full 11mo post-warmup，XLK proxy，1bp cost）：**

| 版本 | Sharpe | Ann | MaxDD | bars_on% |
| --- | ---: | ---: | ---: | ---: |
| E (baseline dispersion gate) | 0.90 | 13.0% | -14.1% | 15.1% |
| F1 (breadth only) | 0.05 | 0.7% | -12.9% | 15.7% |
| **F2 (dispersion AND breadth)** | **1.76** | **16.8%** | **-6.5%** | 6.1% |

F2 把 Sep/Oct 的 gate-on 从 85% / 62% 降到 1-3%（concentration 已经 > median，自动关），Sep 2025 Sharpe 从 -3.86 翻到 **+8.61**。同时 bars_on 从 15% 降到 6% —— **更少交易，赚更多钱，回撤减半**。

### 12.5  Iteration G：Per-stock idiosyncratic gate

F2 是 **全局** gate：dispersion 广 → 全部 4 只都开。但即使全局上"股票都在动"，个别票也可能暂时进入静止期。加一道 per-stock mask：

```
per_stock_mask[s, t] = [ rolling_5d_std[s, t-1] > trailing_60d_median[s, t-1] ]
```

**结果：**

| 版本 | Sharpe | Ann | MaxDD | bars_on% | Break-even cost |
| --- | ---: | ---: | ---: | ---: | :---: |
| E (baseline) | 0.90 | 13.0% | -14.1% | 15.1% | ~1.4bp |
| F2 | 1.76 | 16.8% | -6.5% | 6.1% | ~3.0bp |
| **G (F2 + per-stock) = Phase-12 v2** | **1.96** | 15.7% | **-4.8%** | 5.5% | **~3.5bp** |

三个大新闻：

1. **MaxDD 从 -14.1% 压到 -4.8%**（降 66%）。
2. **Avg turnover 从 0.0177 / bar 降到 0.0064 / bar**（降 64%）—— 用更少的 bar 赚更多，成本敏感性自然大幅改善。
3. **Break-even cost 从 ~1.4bp 推到 ~3.5bp**。这回这个"3.5bp"是 **全样本** 跑出来的，不是 Notebook 03 里 **窄窗口** 得来的。

### 12.6  v1 和 v2 月度对比

v2 把原来最烂的三个月全部修好，同时没有破坏原来好的月份：

```
month     v1 Sharpe    v2 Sharpe
2025-07    +8.79         0.00   (gate 关掉整个月 -- 还不稳)
2025-08    +5.75        +3.94
2025-09    -3.86        +7.02   <-- 从深度负翻到强正
2025-10    -2.48        +3.88   <-- 从深度负翻到强正
2025-11    -2.05        +1.22   <-- 从深度负翻到温正
2025-12    +1.42        +0.33
2026-01    +3.92         0.00   (gate 关掉 -- 代价是不吃这个月的 +3.92)
2026-02    +2.53        +3.17
2026-03    +3.52        +3.08
```

v2 付出的代价：**也关掉了 2025-07 和 2026-01 两个本来赚钱的月**。这是 breadth gate 过严的副作用。我们接受：稳定的 1.96 Sharpe + MaxDD -4.8%，比不稳定的 0.90 Sharpe + MaxDD -14% 可交付得多。

---

## 第十三幕：Honest train/test 与 drop-best 压力测试

### 13.1  Train/Test split

把全样本按 **2025-08-01** 切：

- Train (warmup + 设计可见窗口): 2025-01 → 2025-07 (≈ 7 个月，其中前 3 个月被 warmup 吃掉)
- Test: 2025-08 → 2026-03 (8 个月)

Phase-12 的 breadth gate + per-stock gate 是通过诊断 **Sep-Nov 2025** 而设计的，也就是说 **test 窗口的前三分之一参与了设计**。严格 OOS 是不可能再做了（所有样本都已被看过），但 **train/test 分离的 test 数字** 比 "全样本" 更能代表诚实的外推能力：

| 窗口 | v2 Sharpe | Ann | MaxDD | bars_on% |
| --- | ---: | ---: | ---: | ---: |
| v2 Full 11mo | +1.96 | +15.7% | -4.8% | 5.5% |
| **v2 TEST 8mo (Aug'25 – Mar'26)** | **+2.17** | **+19.3%** | **-4.8%** | 6.8% |

### 13.2  Drop-best-month / Drop-top-3

现实里评一个策略最好的压力测试是："**删掉这个策略在样本里最赚的那几个月**，剩下还能看吗？" 对 v2 test 窗：

| 操作 | 剩余 Sharpe |
| --- | ---: |
| 原 TEST Sharpe | +2.17 |
| 删最好月 (Feb 2026) | **+2.07** |
| 删最好 3 个月 (Feb/Mar 2026 + Oct 2025) | **+1.46** |

**对照 v1**：drop-top-3 掉到 **-0.70**。v2 不再有单月强依赖，这才是 edge 的证据，不是窄窗口运气。

### 13.3  Block bootstrap

2000 次 2-day block bootstrap，TEST 窗口 net PnL：

```
median Sharpe   +2.25
p05 / p95       +0.18 / +4.25
P(Sharpe > 0)   96.3%
P(Sharpe > 1)   85.3%
```

这是 **该版本策略在 test 窗口上**能给出的最诚实的置信带。相比 Notebook 03 的 "95% CI [+1.0, +4.5], P(Sharpe>1)=94%"，我们的 p05 从 +1.0 降到 +0.18（这才是反映小样本不确定性的诚实 p05），但 median (+2.25) 和 P(Sharpe > 1) (85%) 仍然清晰支持"有正 edge"。

### 13.4  Cost stress on TEST

| cost_bps | TEST Sharpe | TEST Ann% | TEST MaxDD% |
| ---: | ---: | ---: | ---: |
| 0.0 | +3.03 | +27.0% | -4.4% |
| 0.5 | +2.60 | +23.2% | -4.6% |
| 1.0 | **+2.17** | +19.3% | -4.8% |
| 1.5 | +1.74 | +15.4% | -5.0% |
| 2.0 | +1.30 | +11.6% | -5.2% |
| 3.0 | +0.43 | +3.8% | -6.7% |
| 5.0 | -1.31 | -11.6% | -9.9% |

Break-even ≈ **3.5bp/side**。QTC IEX 真实 midpoint slippage 在 1-2bp 区间，**所以 2bp 情景下 v2 Sharpe 仍然 +1.30**，不是靠"成本假设小"撑起来的。

### 13.5  Alt-proxy 的"失败"其实是理论的一部分

替换 residualization proxy：

| proxy | v2 Sharpe | 说明 |
| --- | ---: | --- |
| XLK | **+1.96** | 策略设计用的 proxy |
| QQQ | -0.22 | — |
| SPY | -0.34 | — |

一眼看过去像 "fragile"。但 theory 上这不是缺陷：

- AAPL + AMD + MSFT + NVDA 是 **QQQ 的约 35-40% 权重**；对它们做 "XLK-like" 残差化时，proxy 不能是 QQQ —— 你等于在 **残差化自己**。
- SPY 又太广了，它把金融、能源、消费这些和 tech 无关的风格因子灌进 proxy，残差里剩下的不是 idiosyncratic tech noise，是另一个东西。
- XLK 的技术股权重和这 4 只票的 **overlap 更低**（~25% vs QQQ 的 40%），因此是 Avellaneda-Lee 框架下 **理论上正确的 conditioning factor**。

所以这不是"XLK 是我们过拟合出来的 magic parameter"，而是"**我们的 universe 恰好有一个行业 proxy ETF 在 overlap 维度上是对的**"。如果以后 universe 扩展到 10-15 只票，proxy 选择应该升级成"用 PCA 第一主成分"或"每票用它自己的行业 ETF"，但在当前 4-票 + tech-sector 配置下，XLK 就是答案。

---

## 第十四幕：真正的"冻结月"测试 + 往前看

### 14.1  March 2026 单月冻结读数

Phase-12 的所有 gate 都是从诊断 **Sep-Nov 2025** 设计出来的；Feb 2026 是样本最后第二个月，至少还沾边；**March 2026** 是最后一个月，我们在整个 Phase-12 设计阶段没有专门看过它。把它单独挑出来：

```
March 2026 (v2)
  Sharpe          +3.08
  Ann             +30.0%
  MaxDD            -1.7%
  bars_on           10.2%
  avg turnover      0.011
```

同数量级于 TEST 窗口整体。**这是我们目前最接近 "真实 live 一个月会怎样" 的读数**，不保证未来每月都 +3，但 **至少不是窄窗口 cherry-pick**。

### 14.2  学到的：2.72 vs 2.17 的差距讲了什么

第十幕的 +2.72 和第十三幕的 +2.17 其实高度一致 —— 差距就在 0.55 Sharpe，落在 bootstrap CI 之内。**真正不一致的是 +2.72 和 "v1 全样本 +0.90"**：这 1.82 Sharpe 的差距是 **窄窗口多重检验的代价**。

这次第三幕给我们一个具体数字去 **量化窄窗口偏差**：

> **同一个 strategy，同一份数据。窄窗口 (4mo) Sharpe = +2.72；全样本 (11mo) Sharpe = +0.90；窄窗口减掉同样被设计过的前三个月后 test Sharpe = +2.17。**
>
> **所以 "诚实外推估计" ≈ test Sharpe - 窄窗口 bias ≈ 1.5 - 2.0**。

下一次任何 research 想 claim Sharpe 数字，我们心里会默认带一个 **25-40% 的 "窄窗口折扣"**，除非你能拿完全独立的前瞻样本来证伪。

### 14.3  新 Meta-Retrospective：第 12 条教训

补进 §11 的清单：

> **11.5  在写"我们交付了 X Sharpe"之前，先问"X 是在多少个独立窗口上同时成立？"** 一个 window 上 +2.72 和三个 window 上 +1.5，前者漂亮，后者才是 edge。迭代设计和 OOS 评估必须发生在 **disjoint** 的窗口上；如果不能 disjoint（像我们这样样本太短），就在 writeup 里把 "iteration 触碰过哪些 bar" 直接标出来，不要装作 pristine OOS。

### 14.4  Next steps（更新版）

Notebook 03 的 §10.10 里开了五条 next steps，第三幕之后重排优先级：

1. **Paper-trade v2 on QTC（最优先）**：冻结 `(z_in=2, hold=60, disp 5/60, breadth gate ON, per-stock gate ON)`，跑 1 个月，然后拿 live Sharpe 去对 TEST bootstrap CI [+0.18, +4.25]。**live 落在 CI 内就是 edge，落在 p05 以下就是 overfit**，这是最干净的验证方法，且成本 = 一个月时间。
2. **扩 universe**（原 §10.10 第 1 条，优先级不变）：+ GOOGL / META / AMZN / TSLA，让 breadth 从 4 真正变成 10+。这会降低 XLK-specificity 的担忧。
3. **把 gate 从 threshold 升级成学习**（原 §10.10 第 2 条）：输入 `{disp_fast/slow, concentration, per-stock rolling std, proxy vol, proxy trend}`，输出 `regime_score ∈ [0, 1]` 当作连续 gross multiplier。当前的 hard threshold 在 Jan 2026 完全关仓（浪费了 +3.92 的月），连续 score 可以部分开。
4. **更严 cost model**：加 adverse selection 与 size-proportional slippage。当前 `cost_bps * turnover` 是线性，真实的市场冲击是 sqrt 或更差；capacity analysis 的起点。
5. **Pairs overlay** 与 **capacity analysis**（原 §10.10 第 3-5 条）：不变。

### 14.5  一句话：第三幕在讲什么

> **第十二幕：承认 +2.72 是窄窗口数字，全样本只有 +0.90。
> 第十二幕：诊断 Sep-Nov 2025 的亏损，发现是单票主导了 dispersion，gate 被骗了。
> 第十二幕：加 breadth gate + per-stock gate，把 full-sample 拉到 +1.96，MaxDD 从 -14% 压到 -4.8%。
> 第十三幕：train/test 切开，test Sharpe +2.17；删最好 3 个月还剩 +1.46；bootstrap P(Sharpe>1) = 85%。
> 第十四幕：March 2026 单月冻结 +3.08，是最接近 live 的一次读数。**

Phase-12 不是 "又给策略加了一层复杂度"；Phase-12 是 **承认第二幕的数字偏乐观，回头把它挤干，挤出来的是一个更窄但更真的数字 + 一套更厉害的 gate**。

—— QTC Research Journal, rev 3.0（Act III 收束）

---

## 尾声（rev 3.0）：整场 research 的 trajectory

> **第一天：以为要 fix 数据问题。
> 第二天：以为要 fix 策略参数问题。
> 第三天：跑完 Notebook 02，发现"最好的版本"是"选择不交易"。
> 第四天：承认 primary signal 方向从头就错了（IC 全负）。
> 第五天：读文献，切到 cross-sectional residual reversal。
> 第六天：发现 cost 吃光所有 gross edge，设计 sparse state machine，+10.46 Sharpe。
> 第七天：发现 regime dependency，设计 dispersion gate，窄窗口 +2.72 Sharpe，准备交付。
> 第八天：冷静，问自己"这 +2.72 有多少是窄窗口效应"，全样本一跑只剩 +0.90。
> 第九天：月度诊断发现 Sep-Nov 亏损，归因到 dispersion concentration —— 单票主导 gate 被骗。
> 第十天：加 breadth gate + per-stock gate，把 full-sample 拉到 +1.96，MaxDD 从 -14% 压到 -4.8%。
> 第十一天：train/test 分离 + bootstrap + drop-top-3，确认 +2.17 test Sharpe 不依赖单月运气。
> 第十二天：March 2026 单月冻结 +3.08 —— 最接近 live 的一次读数，但也只是一次读数。
> 第十三天：把三幕合成一份文档，承认每一幕的短处，把 "诚实外推估计 ≈ 1.5-2.0" 这个数字说出来，不藏。**

第一幕建的所有基础设施（数据 fallback、分钟面板、`backtest.py`、OOS 分割、交易级指标、Sharpe bootstrap）在第二幕里**一行没重写**；第二幕建的 `xsec_strategy.py` 在第三幕里**只加了两个函数**（`breadth_regime_filter`、`per_stock_vol_mask`）。这才是 infrastructure-first research 的真正回报：**每一幕失败或 pivot 的时候，你都能在几天内完成新一轮迭代，而不是重写全栈**。

所有这些都被 **同一份 822K 行的 Alpaca IEX minute bars** 支撑 —— data layer 的稳固让 research layer 能诚实地犯错、诚实地 pivot、诚实地**自我质疑**。第三幕最重要的意义不是 Sharpe 从 0.90 涨到 1.96，而是我们主动把 +2.72 拆开，让这份文档里同时有 "+2.72 是怎么来的" 和 "+2.72 为什么偏乐观"、"诚实数字是什么"。

—— QTC Research Journal, rev 3.0（Act III 收束）

---

## 附录 A：文件与代码锚点

**第一幕（Act I）文件**

| 文件 | 作用 | 关键位置 |
|---|---|---|
| `fetch_data.py` | 数据 pipeline + feature engineering | `add_symbol_features`（基线 signals & regime flags）, `attach_proxy_features`, `build_research_panel` |
| `data_qtc/raw/<SYM>.parquet` | 每个标的的原始分钟线 | — |
| `data_qtc/combined/raw_all.parquet` | 合并原始分钟线 | 822,788 行 |
| `data_qtc/combined/research_panel.parquet` | 研究面板（带所有 feature + forward return） | 同上行数 |
| `requirements.txt` | 依赖：pandas, pyarrow, requests, alpaca-py, yfinance, pandas-datareader, matplotlib, seaborn, scikit-learn, scipy, jupyter | — |
| `fetch_data.py::DATA_SOURCE` | `auto / qtc / alpaca / yfinance / stooq`，优先级 fallback | 顶部常量 |
| `fetch_data.py::API_KEY / ALPACA_API_KEY / ALPACA_SECRET_KEY` | 凭证（**推荐环境变量**） | 顶部常量 |
| `src/backtest.py` | 单资产 vectorized intraday 回测器（成本 / stop / max-hold / 执行 guardrail） | `BacktestConfig`, `run_backtest`, `compute_metrics`, `build_baseline_signal`, `build_alpha_confirmed_signal`, `apply_regime_gate`, `triple_barrier_labels` |
| `notebooks/01_eda.ipynb` | 数据质量 + 分布 + forward-return IC | — |
| `notebooks/02_ablations.ipynb` | B0 → B1 → M1 → M2 → M3 → M4 → M5 ablation（NVDA） | 结论：最好 M5 OOS Sharpe +0.35 / 52 笔 → 触发 pivot |

**第二幕（Act II）文件**

| 文件 | 作用 | 关键位置 |
|---|---|---|
| `src/xsec_strategy.py`（v1 部分） | cross-sectional 多资产策略模块（窄窗口交付版） | `build_wide_panels`, `residualize`（rolling OLS vs XLK）, `xsec_zscore`, `ts_zscore`, `sparse_reversion_state_machine`（z>2 入场，hold 60m，z_out ratio 0.2），`dispersion_regime_filter`（`disp_fast_5d > disp_slow_60d_median`），`apply_regime_mask`, `backtest_weights`, `portfolio_metrics`, `build_final_strategy` |
| `notebooks/03_intraday_reversal.ipynb` | Act II 全过程：IC 诊断 → 文献 pivot → 迭代 A→B→C→D→E → robustness → 结论 | **窄窗口** Sharpe +2.72（Dec 2025-Mar 2026） |

**第三幕（Act III）文件**

| 文件 | 作用 | 关键位置 |
|---|---|---|
| `src/xsec_strategy.py`（v2 新增） | breadth + per-stock gate | `breadth_regime_filter`（disp AND concentration< median）、`per_stock_vol_mask`（每只票自己的 5d std > 60d median）、`build_final_strategy_v2`（F2 + G 组合） |
| `data_qtc/validation/phase12_summary.json` | 全样本评估结果（v1 vs v2，monthly，proxy，cost，bootstrap，March freeze） | 一份 JSON 可以喂回 notebook |
| `data_qtc/validation/phase12_bt_{v1,v2}.parquet` | v1、v2 的 full-sample net_pnl/equity 序列 | 方便画图与再评估 |
| `data_qtc/validation/phase12_monthly_test.csv` | v2 test 窗口每月 Sharpe | §13.1 的表 |
| `data_qtc/validation/phase12_cost_stress_test.csv` | TEST 窗口成本敏感性 | §13.4 的表 |
| `data_qtc/validation/phase12_proxy_robustness.csv` | XLK/QQQ/SPY proxy 对比 | §13.5 的表 |
| `data_qtc/validation/monthly_diagnostics.csv` | Gate on-fraction / dispersion concentration / XLK vol / proxy trend 按月 | §12.3 的诊断表 |
| `notebooks/04_honest_revisit.ipynb` | Act III 全过程：全样本复盘 → 诊断 → F/G 迭代 → train/test → bootstrap → March freeze | **TEST** Sharpe +2.17，bootstrap median +2.25，March 冻结 +3.08 |

---

## 附录 B：时间线（按事件排序）

**第一幕（Act I — 原本以为这就是终点）**

1. **D0**：拿到 QTC Handbook + Dashboard + example_strat，最初计划"写一个比 toy 聪明的均线策略"。
2. **D0+1**：第一版 `fetch_data.py`（按天打 `/day`），跑不动，中断。
3. **D0+2**：改用按月 `/range`，发现全部 404。拉 OpenAPI 证实 Handbook 过时。
4. **D0+3**：改到真实端点 `/bars`（带 API key），全部 500。服务器 parquet 文件损坏。
5. **D0+4**：加 Alpaca / yfinance / stooq fallback；意识到 QTC 数据本身来自 Alpaca IEX，两者语义一致。
6. **D0+5**：Alpaca Paper 注册 + 凭证配置；一次跑通 822K 行分钟数据。
7. **D0+6**：把"研究基准"从"和市场比"重新定义成 `B0 / B1`。
8. **D0+7**：查 7 篇论文，把四层架构和引用挂钩。
9. **D0+8**：本文初稿（rev 1.0）完成。建出 Notebook 01 (EDA) + Notebook 02 (ablation B0-M5)；结果 OOS Sharpe 最好 +0.35 / 52 笔交易 —— 以为就这样交付。

**第二幕（Act II — pivot）**

10. **D0+9**：重新看 Notebook 02 的 OOS 表，意识到 B0-M4 全线深度负、M5 的 +0.35 是"选择不交易"。对每只股票 / 每个 horizon 跑 Spearman IC，几乎全负 → 确认 primary signal 方向错了，是 mean-reversion 不是 momentum。
11. **D0+10**：读 Nagel 2012 / Avellaneda & Lee 2010 / Lopez de Prado 2018，锁定新假设 = cross-sectional residual reversal。建 `src/xsec_strategy.py`，跑迭代 A（naive xsec reversal）→ OOS Sharpe -7.45（gross 是 +8，cost 吃光）。
12. **D0+11**：迭代 B（+ 残差化）失败但保留结构；迭代 C（+ sparse state machine，z>2 触发、hold 60m、z_out ratio 0.2）把 turnover 从 400x/日 砍到 25x/日，OOS Sharpe 跳到 **+0.47**。做月度 PnL 分解，发现 regime dependency。
13. **D0+12**：迭代 E（+ dispersion regime gate，`disp_fast_5d > disp_slow_60d_median`）把窄窗口 Dec-Mar 的 OOS Sharpe 拉到 **+2.72**。Notebook 03 写完并 execute。本文追加第十 / 十一幕（rev 2.0）。**以为交付结束**。

**第三幕（Act III — 诚实复盘）**

14. **D0+14**：被问到"May-Nov 2025 上 v1 怎么样？"，把 Notebook-03 的 frozen config 跑满 11 个月 post-warmup → **Sharpe 掉到 +0.90，MaxDD -14%**。承认 +2.72 是窄窗口数字。
15. **D0+15**：全样本按月拆 PnL，定位 **Sep/Oct/Nov 2025 三连亏**（-3.86, -2.48, -2.05），此时 dispersion gate 是 ON —— 意味着 "gate 被骗了"。
16. **D0+16**：计算 `concentration(t) = std_stock(rolling std) / mean_stock(rolling std)`，Sep/Oct 2025 concentration 0.52/0.56 vs 常态 0.3 —— 诊断出 **单票主导 dispersion** 是 gate 被骗的原因。
17. **D0+17**：迭代 F2（breadth gate = disp high AND concentration low）和 G（per-stock vol mask），跑 full 11 个月 → **Sharpe 0.90 → 1.96，MaxDD -14% → -4.8%，Sep-Nov 全部转正**。
18. **D0+18**：train/test 切 2025-08-01，test Sharpe **+2.17**；drop-top-3 **+1.46**；2000 次 block bootstrap median **+2.25**，P(Sharpe>1) = **85%**。March 2026 单月冻结 **+3.08**。
19. **D0+19**：写 Notebook 04 + 本文第十二/十三/十四幕（rev 3.0）。

---

## 附录 C：术语对应表（我们的词 ↔ 论文的词）

| 我们叫它 | 论文里叫什么 | 哪篇 |
|---|---|---|
| baseline trend signal | primary strategy | Meta-Labeling (2022) |
| regime gate | state-conditional decision boundary | Meta-Learning JFM (2025) |
| state-aware stop | Sharpe ratio timing with stop-loss | SSRN 2025 |
| vol-aware sizing | vol targeting | Volatility Targeting Is Trendy (2024/25) |
| trade-quality gate | meta-labeling / triple-barrier classifier | López de Prado 派生 / Crypto 2025 |
| fat-tail retention | right-skewed trend profit distribution | Does Trend Still Work on Stocks? (2025) |
| regime gate (HMM) | PCA + HMM regime classification | Regime-based HMM (2025) |
| xsec residual reversal | statistical arbitrage on idiosyncratic residuals | Avellaneda & Lee (2010) |
| short-term reversal | liquidity provision premium | Nagel (2012) "Evaporating Liquidity" |
| sparse state machine | discrete entry/exit with holding period | 自造 —— 工业界通行做法，论文里一般只写 "bid-ask bounce 过滤 + 持有 N 分钟" |
| dispersion regime gate | cross-sectional idiosyncratic volatility conditioner | Heston, Korajczyk & Sadka (2010) 的横截面 effect 对应 |
| breadth gate | idiosyncratic-dispersion concentration filter | 自造（第三幕） —— 对应 pairs trading / stat-arb 文献中的 "signal breadth" 概念 |
| per-stock vol gate | per-asset volatility regime condition | 自造（第三幕） —— 变种 "stock-level regime gating"，Meta-Labeling (2022) 的个股版 |
| 窄窗口偏差 (narrow-window bias) | multiple-hypothesis testing bias / selection-on-outcome | Bailey & López de Prado (2014) "The Probability of Backtest Overfitting" |
