from OptionsBT import OptionsBT
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from TypeEnums import InstrumentType

def bank_nifty_daily_straddle_current_day():
    op = OptionsBT()
    op.bank_nifty_daily_straddle_for_last_n_days(2)
    return op

def bank_nifty_daily_straddle_n_days(ndays=1000):
    op = OptionsBT()
    op.bank_nifty_daily_straddle_for_last_n_days(ndays)
    return op

def bank_nifty_long_iron_butterfly(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.LONG)
    return op

def bank_nifty_shoft_iron_butterfly(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.SHORT)
    return op

def bank_nifty_calendar_spread_expiry_day_only(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.calendar_spread_straddle_only_on_expiry_day('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100,n_expiry=n_expiry)
    return op

def bank_nifty_calendar_spread(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.calendar_spread_straddle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100,n_expiry=n_expiry)
    return op

def e2e_nifty_strangle(num_expiry, price):
    op = OptionsBT()
    op.e2e_strangle('NIFTY', InstrumentType.IndexOptions, 75, 10, 10000, 100, price=price, n_expiry=num_expiry)
    return op

def e2e_banknifty_strangle(num_expiry, num_lots, price):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.e2e_strangle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=5, margin_per_lot=65000, stop_loss=10000, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op

def e2e_banknifty_long_straddle(cls, num_expiry, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.LONG)
    return op
    
def e2e_banknifty_short_straddle(cls, num_expiry, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss - 1000
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.SHORT)
    return op

op = bank_nifty_calendar_spread_expiry_day_only(n_expiry=104, num_lots=5)