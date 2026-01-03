"""
端到端演示（在线推理 + 真数据接入选项）

环境变量（可选）：
- THEGRAPH_URL        (e.g. Uniswap subgraph)
- CEX_ID              (default: binance)
- CEX_API_KEY / CEX_API_SECRET
- MODEL_PATH           (path to saved ML model joblib)

若未配置，将自动回退到 Mock / Demo 模式。
"""

# -------------------------------------------------
# Path bootstrap (工程级，必须保留)3
# -------------------------------------------------
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# -------------------------------------------------
# Standard libs
# -------------------------------------------------
import os
import json
from datetime import datetime, timezone

# -------------------------------------------------
# Domain models
# -------------------------------------------------
from domain.models.market_state import (
    Action,
    StrategySnapshot,
    StrategyCase,
    CapitalFeatures,
    PoolObservation,
    PoolFeatures,
    MarketContext,
    PortfolioState,
)

# -------------------------------------------------
# Market / Fetchers
# -------------------------------------------------
from market.adapters.market_feed import (
    OnChainFetcher,
    PriceFetcher,
    CEXFetcher,
    build_market_context,
)

# -------------------------------------------------
# Strategy / Risk / Execution
# -------------------------------------------------
from strategy.implementations.rule_strategy import DecisionEngine
from risk.implementations.basic_risk import RiskManager
from execution.adapters.mock_broker import ExecutionEngine

# =================================================
# Load ML model (optional)
# Lazy import: only load MLDecisionModel if model file exists
# This prevents requiring ML dependencies in pure mock mode
# =================================================
ml_model = None
model_path = os.environ.get("MODEL_PATH", "ml_decision_model.joblib")

if os.path.exists(model_path):
    try:
        # Lazy import: only import when model file exists
        from ml.inference import MLDecisionModel
        ml_model = MLDecisionModel.load(model_path)
        print(f"[ML] Loaded model from {model_path}")
    except Exception as e:
        print("[ML] Failed to load model, fallback to rule-based:", e)
        ml_model = None
else:
    print(f"[ML] No model found at {model_path}, using rule-based strategy")

# =================================================
# Configure fetchers
# =================================================
price_fetcher = PriceFetcher()

THEGRAPH_URL = os.environ.get("THEGRAPH_URL")
onchain = OnChainFetcher(THEGRAPH_URL) if THEGRAPH_URL else None

CEX_ID = os.environ.get("CEX_ID", "binance")
CEX_KEY = os.environ.get("CEX_API_KEY")
CEX_SECRET = os.environ.get("CEX_API_SECRET")

try:
    cex = CEXFetcher(
        exchange_id=CEX_ID,
        api_key=CEX_KEY,
        api_secret=CEX_SECRET,
    )
except Exception as e:
    print("[CEX] Not configured or init failed, execution disabled:", e)
    cex = None

# =================================================
# Helpers
# =================================================
def fetch_pool_obs(pool_id: str) -> PoolObservation:
    if onchain:
        try:
            return onchain.fetch_pool(pool_id)
        except Exception as e:
            print("[OnChain] fetch failed, fallback to mock:", e)

    # Mock fallback
    return PoolObservation(
        pool_id=pool_id,
        dex="uniswap",
        token_a="USDT",
        token_b="USDC",
        apy_current=0.082,
        apy_1h_avg=0.091,
        apy_24h_avg=0.061,
        swap_count_1h=183,
        fee_1h_usd=312.4,
        tvl_usd=48_000_000.0,
        tvl_change_1h_pct=0.8,
        tvl_change_24h_pct=6.2,
    )


def compute_pool_features(
    obs: PoolObservation,
    market: MarketContext,
) -> PoolFeatures:
    swaps = max(1, obs.swap_count_1h)
    fee_per_swap = obs.fee_1h_usd / swaps

    gas_usd = (
        market.gas_price_gwei
        * 1e-9
        * 150_000
        * (market.native_price_usd or 1.0)
    )

    fee_to_gas_ratio = fee_per_swap / max(gas_usd, 1e-9)

    denom = max(1e-4, abs(obs.apy_24h_avg))
    raw = (obs.apy_current - obs.apy_24h_avg) / denom
    clipped = max(-3.0, min(3.0, raw))
    rel_rank = (clipped + 3.0) / 6.0

    return PoolFeatures(
        pool_id=obs.pool_id,
        fee_to_gas_ratio=fee_to_gas_ratio,
        relative_apy_rank=rel_rank,
        apy_trend_3h=obs.apy_current - obs.apy_1h_avg,
        apy_trend_12h=obs.apy_current - obs.apy_24h_avg,
        swap_trend=obs.tvl_change_1h_pct / 100.0,
        fee_trend=0.0,
        tvl_outflow_rate=obs.tvl_change_1h_pct / 100.0,
    )

# =================================================
# Main
# =================================================
def main():
    # --- market context ---
    market = build_market_context(
        chain="ethereum",
        price_fetcher=price_fetcher,
        gas_price_gwei=31.2,
        network_index=1.1,
    )

    # --- pool ---
    pool_id = "0x0000000000000000000000000000000000000000"
    pool_obs = fetch_pool_obs(pool_id)
    pool_feat = compute_pool_features(pool_obs, market)

    # --- snapshot ---
    snapshot = StrategySnapshot(
        market=market,
        pools={pool_obs.pool_id: pool_obs},
        pool_features={pool_feat.pool_id: pool_feat},
        capital=CapitalFeatures(
            total_capital_usd=10_000.0,
            utilized_capital_usd=2_000.0,
            free_capital_ratio=0.8,
            pool_return_variance=0.01,
            fee_stability_score=0.8,
            max_drawdown_7d_pct=-0.05,
            switch_success_rate=0.9,
        ),
    )

    # --- engines ---
    decider = DecisionEngine(ml_model=ml_model)
    risk = RiskManager()
    executor = ExecutionEngine()

    # --- decision ---
    action, target, order_params, confidence = decider.decide(snapshot)
    print("[Decision]", action, target, order_params, confidence)

    # --- portfolio ---
    portfolio = PortfolioState(
        positions={},
        used_margin_usd=0.0,
        available_margin_usd=10_000.0,
        total_equity_usd=10_000.0,
    )

    if action != Action.HOLD and order_params:
        amount = order_params.get("amount_usd", 0.0)
        allowed, reason, adjusted = risk.check_order(
            portfolio,
            snapshot.capital.total_capital_usd,
            amount,
            pool_obs.pool_id,
        )
        print("[Risk]", allowed, reason, adjusted)

        if allowed and adjusted > 0 and cex:
            try:
                ticker = cex.fetch_ticker("USDT/USDC")
                side = "BUY" if action == Action.LONG else "SELL"
                order, trade = executor.place_order(
                    symbol="USDT/USDC",
                    side=side,
                    amount_usd=adjusted,
                    ticker=ticker,
                )
                if trade:
                    portfolio = risk.update_positions_with_trade(portfolio, trade)
                    print("[Execution]", trade)
            except Exception as e:
                print("[Execution] skipped:", e)
        else:
            print("[Execution] skipped (risk or no CEX)")
    else:
        print("[Decision] HOLD")

    # --- persist case ---
    case = StrategyCase(
        snapshot=snapshot,
        decision=action.value,
        target_pool_id=target,
        confidence=confidence,
        expected_edge=0.0,
        created_at=datetime.now(timezone.utc),
    )

    mongo_doc = case.to_mongo_dict(
        key_style="camel",
        as_bson=False,
        extended_json=True,
    )

    print("\n[StrategyCase JSON]")
    print(json.dumps(mongo_doc, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
