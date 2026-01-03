# strategy.py
"""
DecisionEngine: 支持 ML 推理与规则回退，统一返回 Action enum 与 order parameters。
- 如果模型可用且置信度超过阈值，使用 ML 决策；
- 否则回退到可解释规则（legacy rules）。
"""
from typing import Tuple, Optional, Dict
from domain_models import StrategySnapshot, Action
from ml_model import MLDecisionModel

# 小型规则回退（保留之前逻辑）
DEFAULT_PARAMS = {
    "min_free_capital_ratio": 0.05,
    "min_fee_to_gas": 1.2,
    "enter_relative_apy": 0.65,
    "exit_relative_apy": 0.35,
    "max_position_perc_of_capital": 0.2,
    "ml_confidence_threshold": 0.6,  # 若 ML 置信度 >= 阈值则采纳
}

class DecisionEngine:
    def __init__(self, ml_model: Optional[MLDecisionModel] = None, params: dict = None):
        self.ml_model = ml_model
        self.params = params or DEFAULT_PARAMS

    def decide(self, snapshot: StrategySnapshot) -> Tuple[Action, Optional[str], Dict, float]:
        """
        返回 (action: Action, target_pool_id, order_params, confidence)
        order_params: dict {"side": "BUY"/"SELL", "price": None, "amount_usd": ...}
        """
        # try ML first (if available)
        if self.ml_model and self.ml_model.is_fitted:
            try:
                action, conf = self.ml_model.predict(snapshot)
                if conf >= self.params["ml_confidence_threshold"]:
                    # map Action to order side & amount (simple mapping)
                    side = "BUY" if action == Action.LONG else "SELL"
                    # amount: based on capital & relative_apy_rank if available
                    # fallback fixed
                    amount = snapshot.capital.total_capital_usd * 0.05
                    return action, None, {"side": side, "price": None, "amount_usd": amount}, conf
            except Exception as e:
                # failed ML => fallback to rules
                print("ML predict failed, fallback to rules:", e)

        # RULE-BASED fallback
        capital = snapshot.capital
        if capital.free_capital_ratio < self.params["min_free_capital_ratio"]:
            return Action.HOLD, None, {}, 0.0

        best_score = -999
        best_pool = None
        best_pf = None
        for pf in snapshot.pool_features.values():
            score = pf.relative_apy_rank * 0.6 + min(1.0, pf.fee_to_gas_ratio / 5.0) * 0.3 - max(0.0, pf.tvl_outflow_rate) * 0.1
            if score > best_score:
                best_score = score
                best_pool = pf.pool_id
                best_pf = pf

        if best_pool is None:
            return Action.HOLD, None, {}, 0.0

        pf = best_pf
        if pf.relative_apy_rank >= self.params["enter_relative_apy"] and pf.fee_to_gas_ratio >= self.params["min_fee_to_gas"]:
            amount = snapshot.capital.total_capital_usd * min(self.params["max_position_perc_of_capital"], 0.1 + pf.relative_apy_rank * 0.2)
            return Action.LONG, best_pool, {"side": "BUY", "price": None, "amount_usd": amount}, 0.6

        if pf.relative_apy_rank <= self.params["exit_relative_apy"] or pf.tvl_outflow_rate > 0.05:
            amount = snapshot.capital.utilized_capital_usd * 0.2
            return Action.SHORT, best_pool, {"side": "SELL", "price": None, "amount_usd": amount}, 0.5

        return Action.HOLD, None, {}, 0.0