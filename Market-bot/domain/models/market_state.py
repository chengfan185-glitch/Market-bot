# domain_models.py
from dataclasses import dataclass, asdict, is_dataclass, fields
from typing import Dict, Optional, Any, List, Union, get_origin, get_args
from datetime import datetime, timezone
import re
from enum import Enum

# ---------------------------
# Action Enum
# ---------------------------
class Action(str, Enum):
    HOLD = "HOLD"
    LONG = "LONG"    # enter / add (买入/提供流动性 / enter position)
    SHORT = "SHORT"  # exit / hedge / sell

# ---------------------------
# 字段命名转换工具
# ---------------------------
def snake_to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

def camel_to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()

# ---------------------------
# SerializableMixin: 普通 JSON 与 Mongo 导出/回放
# ---------------------------
class SerializableMixin:
    def to_dict(self) -> Dict[str, Any]:
        """标准 JSON 友好序列化（datetime -> ISO Z 字符串）"""
        if is_dataclass(self):
            raw = asdict(self)
        elif isinstance(self, dict):
            raw = dict(self)
        else:
            raise TypeError("to_dict must be called on dataclass or dict")
        return self._serialize(raw, date_as_iso=True)

    def _serialize(self, node, date_as_iso=True):
        if isinstance(node, dict):
            return {k: self._serialize(v, date_as_iso) for k, v in node.items()}
        elif isinstance(node, list):
            return [self._serialize(v, date_as_iso) for v in node]
        elif isinstance(node, datetime):
            if date_as_iso:
                if node.tzinfo is None:
                    node = node.replace(tzinfo=timezone.utc)
                return node.isoformat().replace("+00:00", "Z")
            else:
                return node
        elif isinstance(node, Enum):
            return node.value
        else:
            return node

    def to_mongo_dict(self, key_style: str = "camel", as_bson: bool = True, extended_json: bool = False) -> Dict[str, Any]:
        """
        导出为 Mongo-friendly dict.
        - key_style: "camel" or "snake"
        - as_bson: True -> keep datetime objects (for PyMongo)
        - extended_json: if as_bson=False and extended_json=True, datetime -> {"$date": iso}
        """
        if is_dataclass(self):
            raw = asdict(self)
        elif isinstance(self, dict):
            raw = dict(self)
        else:
            raise TypeError("to_mongo_dict requires dataclass or dict")

        def convert(n):
            if isinstance(n, dict):
                result = {}
                for k, v in n.items():
                    newk = snake_to_camel(k) if key_style == "camel" else k
                    result[newk] = convert(v)
                return result
            if isinstance(n, list):
                return [convert(i) for i in n]
            if isinstance(n, datetime):
                if as_bson:
                    return n
                iso = n.isoformat().replace("+00:00", "Z") if n.tzinfo or n.tzinfo is None else n.isoformat()
                if extended_json:
                    return {"$date": iso}
                else:
                    return iso
            if isinstance(n, Enum):
                return n.value
            return n

        return convert(raw)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从 dict 恢复 dataclass（支持 camel or snake keys 与 {"$date":...}）"""
        if data is None:
            return None
        if not is_dataclass(cls):
            return data
        kwargs = {}
        for f in fields(cls):
            name = f.name
            candidate = None
            if name in data:
                candidate = data[name]
            else:
                camel = snake_to_camel(name)
                if camel in data:
                    candidate = data[camel]
            if candidate is None and candidate != 0:
                continue
            kwargs[name] = cls._deserialize(candidate, f.type)
        return cls(**kwargs)

    @classmethod
    def _deserialize(cls, raw, expected_type):
        if raw is None:
            return None
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # Union / Optional
        if origin is Union:
            non_none = [t for t in args if t is not type(None)]
            for t in non_none:
                try:
                    return cls._deserialize(raw, t)
                except Exception:
                    continue
            return raw

        # datetime -> raw could be {"$date": iso} or iso string or datetime
        if expected_type is datetime:
            if isinstance(raw, dict) and "$date" in raw:
                iso = raw["$date"]
                try:
                    return datetime.fromisoformat(iso.replace("Z", "+00:00"))
                except Exception:
                    return raw
            if isinstance(raw, str):
                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except Exception:
                    return raw
            if isinstance(raw, datetime):
                return raw
            return raw

        # Enum types -> expect str
        if isinstance(expected_type, type) and issubclass(expected_type, Enum):
            if isinstance(raw, str):
                try:
                    return expected_type(raw)
                except Exception:
                    return raw
            return raw

        # list
        if origin in (list, List):
            item_t = args[0] if args else Any
            if isinstance(raw, list):
                return [cls._deserialize(i, item_t) for i in raw]
            return raw

        # dict
        if origin in (dict, Dict):
            val_t = args[1] if len(args) > 1 else Any
            if isinstance(raw, dict):
                out = {}
                for k, v in raw.items():
                    nk = camel_to_snake(k) if re.search(r"[A-Z]", k) else k
                    out[nk] = cls._deserialize(v, val_t)
                return out
            return raw

        # nested dataclass
        try:
            if isinstance(expected_type, type) and is_dataclass(expected_type):
                if isinstance(raw, dict):
                    normalized = {}
                    for k, v in raw.items():
                        nk = camel_to_snake(k) if re.search(r"[A-Z]", k) else k
                        normalized[nk] = v
                    return expected_type.from_dict(normalized)
        except Exception:
            pass

        return raw

# ---------------------------
# 数据模型（与之前保持兼容）
# ---------------------------
@dataclass
class MarketContext(SerializableMixin):
    timestamp: datetime
    chain: str
    gas_price_gwei: float
    network_congestion_index: float
    native_price_usd: Optional[float] = None

@dataclass
class PoolObservation(SerializableMixin):
    pool_id: str
    dex: str
    token_a: str
    token_b: str
    apy_current: float
    apy_1h_avg: float
    apy_24h_avg: float
    swap_count_1h: int
    fee_1h_usd: float
    tvl_usd: float
    tvl_change_1h_pct: float
    tvl_change_24h_pct: float

@dataclass
class PoolFeatures(SerializableMixin):
    pool_id: str
    fee_to_gas_ratio: float
    relative_apy_rank: float
    apy_trend_3h: float
    apy_trend_12h: float
    swap_trend: float
    fee_trend: float
    tvl_outflow_rate: float

@dataclass
class CapitalFeatures(SerializableMixin):
    total_capital_usd: float
    utilized_capital_usd: float
    free_capital_ratio: float
    pool_return_variance: float
    fee_stability_score: float
    max_drawdown_7d_pct: float
    switch_success_rate: float

@dataclass
class StrategySnapshot(SerializableMixin):
    market: MarketContext
    pools: Dict[str, PoolObservation]
    pool_features: Dict[str, PoolFeatures]
    capital: CapitalFeatures

@dataclass
class StrategyCase(SerializableMixin):
    snapshot: StrategySnapshot
    decision: str  # legacy (keeps string), ML will produce Action enum
    target_pool_id: Optional[str]
    confidence: float
    expected_edge: float
    created_at: datetime
    pnl_pct: Optional[float] = None
    duration_hours: Optional[float] = None

# Execution and portfolio models (kept same as before)
@dataclass
class MarketTicker(SerializableMixin):
    symbol: str
    bid: float
    ask: float
    last: float
    volume_24h: Optional[float]
    timestamp: datetime

@dataclass
class Order(SerializableMixin):
    order_id: str
    symbol: str
    side: str
    price: Optional[float]
    amount: float
    status: str
    created_at: datetime
    filled_amount: float = 0.0
    filled_avg_price: Optional[float] = None

@dataclass
class Trade(SerializableMixin):
    trade_id: str
    order_id: str
    symbol: str
    side: str
    price: float
    amount: float
    fee_usd: float
    timestamp: datetime

@dataclass
class Position(SerializableMixin):
    symbol: str
    side: str
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl_usd: float
    realized_pnl_usd: float
    leverage: Optional[float] = 1.0
    updated_at: Optional[datetime] = None

@dataclass
class PortfolioState(SerializableMixin):
    positions: Dict[str, Position]
    used_margin_usd: float
    available_margin_usd: float
    total_equity_usd: float
    max_drawdown_pct: Optional[float] = None