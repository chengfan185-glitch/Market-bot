# execution.py
"""
执行系统（模拟）：
- place_order: 创建 Order, 模拟市场成交（立即按 mid-price 填充或部分成交）
- cancel_order: 更改订单状态
- 返回 Trade 记录（用于回放与 PnL 计算）

在真实环境中：
- 将调用交易所 API (ccxt, rest/ws) 或 AMM tx签名广播到链上（需要私钥管理、nonce、gas）
- 建议：异步下单 + 回调/监听成交回报（WebSocket / On-chain events）
"""

import uuid
from datetime import datetime, timezone
from typing import Tuple, Optional
from domain.models.market_state import Order, Trade, MarketTicker, Position, PortfolioState

FEE_RATE = 0.0006  # 假设手续费率 0.06% per trade (示例)

class ExecutionEngine:
    def __init__(self):
        self.orders = {}  # order_id -> Order

    def _gen_id(self) -> str:
        return str(uuid.uuid4())

    def place_order(self, symbol: str, side: str, amount_usd: float, ticker: MarketTicker, price: Optional[float] = None) -> Tuple[Order, Optional[Trade]]:
        """
        模拟下单，立即以 mid-price 成交（可以扩展滑点、部分成交）
        返回 Order 与 Trade（若成交）
        """
        order_id = self._gen_id()
        created = datetime.now(timezone.utc)
        order = Order(order_id=order_id, symbol=symbol, side=side, price=price or ticker.last, amount=amount_usd, status="NEW", created_at=created)
        # Simulate immediate fill
        executed_price = price or ticker.last
        filled_amount = amount_usd  # 因我们使用 USD 计量下单
        order.status = "FILLED"
        order.filled_amount = filled_amount
        order.filled_avg_price = executed_price
        self.orders[order_id] = order

        # fee (USD) = amount_usd * fee_rate
        fee = filled_amount * FEE_RATE
        trade = Trade(trade_id=self._gen_id(), order_id=order_id, symbol=symbol, side=side, price=executed_price, amount=filled_amount, fee_usd=fee, timestamp=created)
        return order, trade

    def cancel_order(self, order_id: str) -> Optional[Order]:
        o = self.orders.get(order_id)
        if not o:
            return None
        if o.status in ("FILLED", "CANCELED"):
            return o
        o.status = "CANCELED"
        return o