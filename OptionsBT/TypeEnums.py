from enum import Enum

class PositionType(Enum):
    """Represents enums of position types"""
    Long = 1
    Short = 2

class CandleData(Enum):
    """Candle Data enum"""
    OPEN = 1
    HIGH = 2
    LOW = 3
    CLOSE = 4

class InstrumentType(object):
    """description of class"""
    IndexOptions = "OPTIDX"
    IndexFutures = "FUTIDX"
    StockOptions = "OPTSTK"
    StockFutures = "FUTSTK"
    Index = "IDX"
    Stock = "STK"

    @classmethod
    def getfutures(cls, instrument):
        if "IDX" in instrument:
            return cls.IndexFutures
        elif "STK" in instrument:
            return cls.StockFutures
        return None