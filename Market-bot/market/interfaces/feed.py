from abc import ABC, abstractmethod

class MarketFeed(ABC):
    @abstractmethod
    def snapshot(self):
        pass
