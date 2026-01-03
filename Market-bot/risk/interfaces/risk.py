from abc import ABC, abstractmethod

class RiskRule(ABC):
    @abstractmethod
    def check(self, signal):
        pass
