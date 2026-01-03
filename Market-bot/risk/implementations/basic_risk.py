# risk.py
"""
风控模块：
- 在下单前对 order 进行风险校验（仓位限制、最大回撤、单仓占比）
- 提供仓位更新与回撤计算接口
"""

from domain.models.market_state import PortfolioState, Position, Trade
from typing import Tuple
from datetime import datetime, timezone

DEFAULT_RISK = {
    "max_position_pct": 0.25,      # 单仓最大占比
    "max_total_exposure_pct": 0.6, # 组合总暴露最大占比
    "max_drawdown_allowed": 0.25,  # 允许最大回撤 25%
    "min_free_capital_ratio": 0.03 # 下单后至少保留 3% 现金
}

class RiskManager:
    def __init__(self, params: dict = None):
        self.params = params or DEFAULT_RISK

    def check_order(self, portfolio: PortfolioState, capital_total_usd: float, order_amount_usd: float, target_symbol: str) -> Tuple[bool, str, float]:
        """
        检查是否允许下单，返回 (allowed, reason, adjusted_amount_usd)
        - 检查单仓占比、总暴露、下单后最小可用资本
        """
        # 1) 单仓占比限制
        existing_pos = portfolio.positions.get(target_symbol)
        existing_notional = existing_pos.size * existing_pos.mark_price if existing_pos else 0.0
        proposed_total_for_symbol = existing_notional + order_amount_usd
        if proposed_total_for_symbol / capital_total_usd > self.params["max_position_pct"]:
            allowed = False
            reason = "single_position_limit"
            max_allowed = self.params["max_position_pct"] * capital_total_usd - existing_notional
            return False, reason, max(0.0, max_allowed)

        # 2) 总暴露限制 (sum of absolute notionals)
        total_exposure = sum(abs(p.size * p.mark_price) for p in portfolio.positions.values())
        if (total_exposure + order_amount_usd) / capital_total_usd > self.params["max_total_exposure_pct"]:
            return False, "total_exposure_limit", 0.0

        # 3) 最小空余资金
        if portfolio.available_margin_usd - order_amount_usd < self.params["min_free_capital_ratio"] * capital_total_usd:
            # 调整下单量到允许范围
            max_allowed = portfolio.available_margin_usd - self.params["min_free_capital_ratio"] * capital_total_usd
            if max_allowed <= 0:
                return False, "insufficient_free_margin", 0.0
            return True, "partial_allowed", max_allowed

        return True, "ok", order_amount_usd

    def update_positions_with_trade(self, portfolio: PortfolioState, trade: Trade) -> PortfolioState:
        """
        将 Trade 应用到 portfolio（简化：以 USD 为计量单位，side BUY 增持 LONG，SELL 减持）
        """
        symbol = trade.symbol
        side = "LONG" if trade.side == "BUY" else "SHORT"
        now = datetime.now(timezone.utc)
        existing = portfolio.positions.get(symbol)
        notional = trade.amount
        # simplified PnL handling: add or reduce position
        if existing is None:
            pos = Position(
                symbol=symbol,
                side=side,
                size=notional,
                entry_price=trade.price,
                mark_price=trade.price,
                unrealized_pnl_usd=0.0,
                realized_pnl_usd=0.0,
                updated_at=now
            )
            portfolio.positions[symbol] = pos
        else:
            # if same side: increase average entry (weighted)
            if existing.side == side:
                total_size = existing.size + notional
                if total_size > 0:
                    new_entry = (existing.entry_price * existing.size + trade.price * notional) / total_size
                else:
                    new_entry = trade.price
                existing.entry_price = new_entry
                existing.size = total_size
                existing.mark_price = trade.price
                existing.updated_at = now
            else:
                # opposite side: reduce existing position -> realized pnl estimate
                # simplified: realized = (entry - price) * min(size, notional)
                matched = min(existing.size, notional)
                profit = 0.0
                if existing.side == "LONG":
                    profit = (trade.price - existing.entry_price) * matched
                else:
                    profit = (existing.entry_price - trade.price) * matched
                existing.size = existing.size - matched
                existing.realized_pnl_usd += profit - trade.fee_usd
                if existing.size <= 0:
                    # position closed
                    del portfolio.positions[symbol]
                else:
                    existing.updated_at = now

        # update margins / equity (very simplified)
        portfolio.used_margin_usd = sum(abs(p.size * p.mark_price) for p in portfolio.positions.values())
        portfolio.available_margin_usd = max(0.0, portfolio.total_equity_usd - portfolio.used_margin_usd)
        # compute naive max drawdown placeholder (could store peak equity to compute)
        return portfolio