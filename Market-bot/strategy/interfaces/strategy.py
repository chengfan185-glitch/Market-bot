from abc import ABC, abstractmethod

class Strategy(ABC):
    @abstractmethod
    def decide(self, market_state):
        pass
