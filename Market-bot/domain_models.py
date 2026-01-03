# Compatibility shim: re-export domain models for legacy imports
# This allows `from domain_models import X` to work
from domain.models.market_state import (
    Action,
    StrategySnapshot,
    StrategyCase,
    CapitalFeatures,
    PoolObservation,
    PoolFeatures,
    MarketContext,
    MarketTicker,
    Order,
    Trade,
    Position,
    PortfolioState,
)

__all__ = [
    "Action",
    "StrategySnapshot",
    "StrategyCase",
    "CapitalFeatures",
    "PoolObservation",
    "PoolFeatures",
    "MarketContext",
    "MarketTicker",
    "Order",
    "Trade",
    "Position",
    "PortfolioState",
]
