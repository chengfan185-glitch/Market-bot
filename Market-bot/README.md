```markdown
# 完整系统演示（含：采集 / 策略 / 执行 / 风控 / Mongo 序列化）

概述
- 已实现模块：
  - domain_models.py：核心数据模型 + Mongo 友好序列化（camel/snake、PyMongo datetime、Extended JSON）
  - data_fetchers.py：链上（Subgraph）与 CEX（ccxt）采集示例（当前为 Mock）；给出真实接入建议
  - strategy.py：可解释规则策略（输出 LONG/SHORT/HOLD + 下单建议）
  - execution.py：执行引擎（模拟下单/成交/撤单）
  - risk.py：风控（单仓/组合暴露/最小空余资金校验 + 交易后仓位更新）
  - pipeline_demo.py：端到端集成演示

运行
1. Python 3.8+
2. 保存所有文件到同一目录
3. 运行：
   ```
   python pipeline_demo.py
   ```

真实接入建议（数据源）
- On-chain 池级数据（swap_count, fees, tvl, apy）
  - TheGraph / Subgraph（Uniswap, Sushiswap, Balancer 等）
  - Dune Query（聚合分析）
  - 自建 indexer（基于 RPC-log）
- 价格（native_price_usd）
  - Chainlink Price Feeds（更稳健的 on-chain 喂价）
  - CoinGecko / CoinMarketCap（HTTP API）
- CEX 行情
  - ccxt 库（统一访问 Binance/OKX/Bybit 等）
- 历史统计（mean/std/percentiles）
  - 存在时序 DB（Timescale/Influx）或批量 Parquet / S3 存储供 FeatureEngine 查询

策略与风控要点（公式回顾）
- fee_to_gas_ratio:
  - fee_per_swap_usd = fee_1h_usd / max(1, swap_count_1h)
  - gas_cost_per_swap_usd = gas_price_gwei * 1e-9 * avg_gas_per_swap * native_price_usd
  - fee_to_gas_ratio = fee_per_swap_usd / gas_cost_per_swap_usd
- relative_apy_rank:
  - 优先使用跨池 APY 分位数
  - fallback：((apy_current - apy_24h_avg) / apy_24h_avg) 映射到 [0,1]
- 决策：基于 relative_apy_rank、fee_to_gas_ratio、tvl_outflow_rate 与资本可用性
- 风控：单仓占比、组合总暴露、最小空余资金、最大回撤限制

下一步建议
- 把 Mock Fetcher 替换为真实 TheGraph / Chainlink / ccxt 实现，并把 credential/私钥放在安全 Vault（例如 AWS Secrets Manager）
- 将 ExecutionEngine 替换为真实交易连接（支持异步 WebSocket 回报）
- 为策略增加历史回测与单元测试（pytest）
- 将 Mongo 导出直接连接 PyMongo 或写入 Kafka + Sink

```