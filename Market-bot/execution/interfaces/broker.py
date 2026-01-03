from abc import ABC, abstractmethod

class Broker(ABC):
    @abstractmethod
    def place_order(self, order):
        pass
