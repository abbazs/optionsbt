from OptionsBT import OptionsBT
from TypeEnums import PositionType
from multiprocessing.dummy import Pool as ThreadPool
from itertools import repeat
from TypeEnums import InstrumentType


def bank_nifty_daily_straddle_current_day(num_lots=5):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    op.index_daily_straddle('BANKNIFTY', 40, num_lots, 100, stop_loss, 2)
    return op


def bank_nifty_daily_straddle_n_days(num_lots=5, ndays=720):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    op.index_daily_straddle('BANKNIFTY', 40, num_lots, 100, stop_loss, ndays)
    return op


def bank_nifty_long_iron_butterfly(num_lots=5, n_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                      stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.LONG)
    return op


def bank_nifty_shoft_iron_butterfly(num_lots=5, n_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.iron_butterfly('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss,
                      stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry, position_type=PositionType.SHORT)
    return op


def bank_nifty_calendar_spread_expiry_day_only(num_lots=5, n_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle_only_on_expiry_day('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots,
                                                   margin_per_lot=65000, stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry)
    return op


def nifty_calendar_spread(num_lots=5, n_expiry=104):
    op = OptionsBT()
    stop_loss = (50000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle('NIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=50000,
                                stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry)
    return op


def bank_nifty_calendar_spread(num_lots=5, n_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.calendar_spread_straddle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                                stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=n_expiry)
    return op


def e2e_nifty_strangle(num_lots=5, price=50, num_expiry=104):
    op = OptionsBT()
    stop_loss = (50000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_strangle('NIFTY', InstrumentType.IndexOptions, 75, num_lots=num_lots, margin_per_lot=50000,
                    stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op


def e2e_banknifty_strangle(num_lots=5, price=50, num_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_strangle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                    stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op

def e2e_banknifty_dist_strangle(num_lots=5, dist=3, num_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_dist_strangle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                    stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, dist=dist, n_expiry=num_expiry)
    return op

def expiry_day_banknifty_strangle(num_lots=2, price=50, num_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.expiry_day_strangle('BANKNIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                           stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op


def expiry_day_nifty_strangle(num_lots=2, price=50, num_expiry=100):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.expiry_day_strangle('NIFTY', InstrumentType.IndexOptions, lot_size=40, num_lots=num_lots, margin_per_lot=65000,
                           stop_loss=stop_loss, stop_loss_threshold=stop_loss_threshold, brokerage=100, price=price, n_expiry=num_expiry)
    return op


def expiry_day_hdfcbank_strangle(num_lots=2, price=2, num_expiry=100):
    op = OptionsBT()
    mpl = 120000
    sl = (mpl*num_lots*2)*0.015
    slt = sl * 0.9
    op.expiry_day_strangle('HDFCBANK', InstrumentType.StockOptions, lot_size=500, num_lots=num_lots,
                           margin_per_lot=mpl, stop_loss=sl, stop_loss_threshold=slt, brokerage=100, price=price, n_expiry=num_expiry)
    return op


def e2e_banknifty_long_straddle(num_lots=5, num_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss,
                    stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.LONG)
    return op


def e2e_banknifty_short_straddle(num_lots=5, num_expiry=104):
    op = OptionsBT()
    stop_loss = (65000*num_lots*2)*0.015
    stop_loss_threshold = stop_loss * 0.9
    op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots=num_lots, margin_per_lot=65000, stop_loss=stop_loss,
                    stop_loss_threshold=stop_loss_threshold, brokerage=100, n_expiry=num_expiry, position_type=PositionType.SHORT)
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
    op.prepare_strategy(symbol, InstrumentType.IndexOptions, n_expiry)
    return op


def banknifty_2day_before_double_ratio_spread(r1=3, r2=4, n_expiry=104):
    op = OptionsBT()
    op.double_ratio_spread_days_before_expiry(
        'BANKNIFTY', InstrumentType.IndexOptions, 40, 65000, 300, r1=r1, r2=r2, days_before=2, n_expiry=n_expiry)
    return op


def banknifty_e2e_double_ratio_spread(r1=3, r2=4, n_expiry=104):
    op = OptionsBT()
    op.e2e_double_ratio_spread('BANKNIFTY', InstrumentType.IndexOptions,
                               40, 65000, 300, r1=r1, r2=r2, n_expiry=n_expiry)
    return op


def banknifty_e2e_ratio_write_at_max_oi(num_lots=5, n_expiry=104):
    op = OptionsBT()
    op.e2e_ratio_write_at_max_oi(
        'BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots, 65000, 200, n_expiry=n_expiry)
    return op


def banknifty_e2e_naked_write_at_max_oi(num_lots=5, n_expiry=104):
    op = OptionsBT()
    op.e2e_naked_write_at_max_oi(
        'BANKNIFTY', InstrumentType.IndexOptions, 40, num_lots, 65000, 200, n_expiry=n_expiry)
    return op


def e2e_ITC_long_straddle(num_lots=1, num_expiry=60):
    op = OptionsBT()
    margin = 90000
    sl = (margin*num_lots*2)*0.015
    slt = sl * 0.9
    op.e2e_straddle('ITC', InstrumentType.StockOptions, 2400, num_lots=num_lots, margin_per_lot=margin,
                    stop_loss=sl, stop_loss_threshold=slt, brokerage=100, n_expiry=num_expiry, position_type=PositionType.LONG)
    return op


def e2e_lic_short_strangle(num_lots=1, num_expiry=60, price=5):
    op = OptionsBT()
    margin = 75000
    sl = (margin*num_lots*2)*0.015
    slt = sl * 0.9
    op.e2e_strangle('LICHSGFIN', InstrumentType.StockOptions, 1100, num_lots=num_lots, margin_per_lot=margin,
                    stop_loss=sl, stop_loss_threshold=slt, brokerage=100, price=price, n_expiry=num_expiry)
    return op


if __name__ == '__main__':
    #op = bank_nifty_calendar_spread_expiry_day_only(n_expiry=104, num_lots=5)
    #op = nifty_calendar_spread(n_expiry=24, num_lots=5)
    #op = bank_nifty_calendar_spread(104, 5)
    #op = e2e_banknifty_long_straddle(104, 2)
    # print(dir())
    #banknifty_e2e_double_ratio_spread(4, 5, 106)
    # banknifty_2day_before_double_ratio_spread()
    # banknifty_e2e_ratio_write_at_max_oi(2)
    banknifty_e2e_naked_write_at_max_oi(5)
    e2e_banknifty_strangle(num_expiry=52, num_lots=5, price=35)
