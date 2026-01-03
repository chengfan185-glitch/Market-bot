# data_fetchers.py
"""
数据采集模块（链上 / CEX）：
- 提供 Mock 实现以便本地测试
- 给出真实接入建议（TheGraph, Chainlink, CoinGecko, CCXT）

在真实环境中：
- OnChainFetcher.fetch_pool_observations 应使用 Subgraph (TheGraph) 或自建索引节点拉取 pool-level metrics（swap counts, fees, tvl, apy）
- PriceFetcher.fetch_native_price 应使用 Chainlink Price Feed 或 CoinGecko REST API
- CEXFetcher 可以使用 ccxt 库调用 Binance/FTX/OKX 等公共行情
"""

from datetime import datetime, timezone
from typing import List, Dict
import random

from market.adapters.market_feed import OnChainFetcher, PriceFetcher, CEXFetcher


# --------- OnChainFetcher (Subgraph / RPC) ----------
class OnChainFetcher:
    def __init__(self):
        pass

    def fetch_pool(self, pool_id: str) -> PoolObservation:
        # Mock: 返回稳定的示例数据（本地测试）
        return PoolObservation(
            pool_id=pool_id,
            dex="uniswap",
            token_a="USDT",
            token_b="USDC",
            apy_current=0.082 + random.uniform(-0.01, 0.01),
            apy_1h_avg=0.09,
            apy_24h_avg=0.06,
            swap_count_1h=183 + random.randint(-20, 20),
            fee_1h_usd=312.4 + random.uniform(-40, 40),
            tvl_usd=48_000_000.0 + random.uniform(-1e6, 1e6),
            tvl_change_1h_pct=0.8 + random.uniform(-1.0, 1.0),
            tvl_change_24h_pct=6.2 + random.uniform(-2.0, 2.0)
        )

    def fetch_many_pools(self, pool_ids: List[str]) -> List[PoolObservation]:
        return [self.fetch_pool(pid) for pid in pool_ids]

# --------- PriceFetcher (Chainlink / CoinGecko) ----------
class PriceFetcher:
    def fetch_native_price(self, chain: str) -> float:
        # Mock: 返回示例价格 (ETH)
        # 真实实现示例：Chainlink Price Feed 或 CoinGecko API:
        #   - Chainlink: onchain aggregator contract, requires RPC call + ABI
        #   - CoinGecko: GET https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd
        return 1600.0 + random.uniform(-50, 50)

# --------- CEXFetcher (ccxt) ----------
class CEXFetcher:
    def __init__(self):
        # In real env: import ccxt; self.binance = ccxt.binance()
        pass

    def fetch_ticker(self, symbol: str) -> MarketTicker:
        # Mock: 返回示例ticker
        import random
        mid = 1.0 + random.uniform(-0.002, 0.002)
        return MarketTicker(
            symbol=symbol,
            bid=mid - 0.0001,
            ask=mid + 0.0001,
            last=mid,
            volume_24h=1e6,
            timestamp=datetime.now(timezone.utc)
        )

# Helper to compose MarketContext
def build_market_context(chain: str, price_fetcher: PriceFetcher, gas_price_gwei: float, network_index: float) -> MarketContext:
    return MarketContext(
        timestamp=datetime.now(timezone.utc),
        chain=chain,
        gas_price_gwei=gas_price_gwei,
        network_congestion_index=network_index,
        native_price_usd=price_fetcher.fetch_native_price(chain)
    )