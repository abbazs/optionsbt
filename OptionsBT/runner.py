from OptionsBT import OptionsBT
from TypeEnums import PositionType
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from TypeEnums import InstrumentType

def bank_nifty_daily_straddle_current_day():
    op = OptionsBT()
    op.bank_nifty_daily_straddle_for_last_n_days(2)
    return op

def bank_nifty_daily_straddle_n_days(ndays=10):
    op = OptionsBT()
    op.bank_nifty_daily_straddle_for_last_n_days(ndays)
    return op

def bank_nifty_long_iron_butterfly(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.LONG)
    return op

def bank_nifty_shoft_iron_butterfly(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.SHORT)
    return op

def bank_nifty_calendar_spread_expiry_day_only(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle_only_on_expiry_day('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100,n_expiry=n_expiry)
    return op

def nifty_calendar_spread(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (50000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle('NIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=50000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100,n_expiry=n_expiry)
    return op

def bank_nifty_calendar_spread(n_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100,n_expiry=n_expiry)
    return op

def e2e_nifty_strangle(num_expiry=10, num_lots=5, price=50):
    op = OptionsBT()
    stop_loss = (50000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_strangle('NIFTY', InstrumentType.IndexOptions, 75, num_lots=num_lots, margin_per_lot=50000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op

def e2e_banknifty_strangle(num_expiry=10, num_lots=5, price=50):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_strangle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op

def e2e_banknifty_long_straddle(num_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.LONG)
    return op
    
def e2e_banknifty_short_straddle(num_expiry=10, num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.SHORT)
    return op

def test_double_ratio_spread():
    op = OptionsBT()
    op.prepare_strategy('BANKNIFTY', InstrumentType.IndexOptions, 2)
    st = op.expirys['START_DT'].iloc[0]
    nd = op.expirys['EXPIRY_DT'].iloc[0]
    chain = op.double_ratio_spread_builder(st, nd, 40)
    return chain

def fetch_data_only(symbol='BANKNIFTY', n_expiry=10):
    op = OptionsBT()
    op.prepare_strategy(symbol,InstrumentType.IndexOptions, n_expiry)
    return op

def banknifty_2day_before_double_ratio_spread(r1, r2, n_expiry=52):
    op = OptionsBT()
    op.double_ratio_spread_days_before_expiry('BANKNIFTY', InstrumentType.IndexOptions, 40, 65000, 300, r1=2, r2=3, days_before=2, n_expiry=n_expiry)
    return op

def banknifty_e2e_double_ratio_spread(r1, r2, n_expiry=52):
    op = OptionsBT()
    op.e2e_double_ratio_spread('BANKNIFTY', InstrumentType.IndexOptions, 40, 65000, 300, r1=r1, r2=r2, n_expiry=n_expiry)
    return op

if __name__ == '__main__':
    #op = bank_nifty_calendar_spread_expiry_day_only(n_expiry=104, num_lots=5)
    #op = nifty_calendar_spread(n_expiry=24, num_lots=5)
    #op = bank_nifty_calendar_spread(104, 5)
    #op = e2e_banknifty_long_straddle(104, 2)
    #print(dir())
    banknifty_e2e_double_ratio_spread(4, 5, 106)
    #banknifty_2day_before_double_ratio_spread()