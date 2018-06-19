from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

import dbtables
from log import print_exception
from TypeEnums import PositionType, CandleData, InstrumentType
from Utility import fix_start_and_end_date, get_current_date
import Defaults

class OptionsBT(object):
    def __init__(self):
        try:
            self.db = create_engine('sqlite:///D:/Work/db/bhav.db')
            self.meta_data = MetaData(self.db)
            self.fno_table = Table(dbtables.FNO, self.meta_data, autoload=True)
            self.index_table = Table(dbtables.IINDEX, self.meta_data, autoload=True)
            self.stock_table = Table(dbtables.STOCKS, self.meta_data, autoload=True)
            self.last_DF = None
            self.last_stm = None
            self.losing_streak_counter = 0
            self.results = []
        except Exception as e:
            print_exception(e)

    def get_sql_query_statement(self, table, symbol, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            meta = MetaData(self.db)
            dts = Table(table, meta, autoload=True)
            stm = select(['*']).where(and_(dts.c.TIMESTAMP >= start_date, dts.c.TIMESTAMP <= end_date, dts.c.SYMBOL == symbol))
            return stm
        except Exception as e:
            print_exception(e)
            return None

    def get_spot_data_between_dates(self, symbol, instrument, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)

            if "IDX" in instrument:
                self.last_stm = select(['*']).where(and_(self.index_table.c.TIMESTAMP >= start_date, self.index_table.c.TIMESTAMP <= end_date, self.index_table.c.SYMBOL == symbol))
            else:
                self.last_stm = select(['*']).where(and_(self.stock_table.c.TIMESTAMP >= start_date, self.stock_table.c.TIMESTAMP <= end_date, self.stock_table.c.SYMBOL == symbol))

            self.spot_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP'])
            self.spot_DF.drop([self.spot_DF.columns[0]], axis=1, inplace=True)
            self.spot_DF.sort_values(['TIMESTAMP'], inplace=True)
            self.spot_DF.reset_index(drop=True, inplace=True)

            if "IDX" in instrument:
                return self.spot_DF[self.spot_DF.columns[0:6]]
            else:
                return self.spot_DF[self.spot_DF.columns[-2::2].append(self.spot_DF.columns[0:6]).drop('SERIES')]
        except Exception as e:
            print_exception(e)
            return None

    def get_spot_data_for_today(self, symbol, instrument):
        return self.get_spot_data_between_dates(symbol, instrument, get_current_date())

    def get_spot_data_for_last_n_days(self, symbol, instrument, n_days=0):
        end_date = get_current_date()
        # Add 5 days to n_days and filter only required number days
        start_date = end_date - timedelta(days=n_days + 5)
        spot_data = self.get_spot_data_between_dates(symbol, instrument, start_date, end_date)
        return spot_data.tail(n_days)

    def get_fno_data_between_dates(self, symbol, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            self.last_stm = select(['*']).where(and_(self.fno_table.c.TIMESTAMP >= start_date, self.fno_table.c.TIMESTAMP <= end_date, self.fno_table.c.SYMBOL == symbol))
            self.fno_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP', 'EXPIRY_DT'])
            self.fno_DF.sort_values(['OPTION_TYP', 'EXPIRY_DT', 'TIMESTAMP', 'STRIKE_PR'], inplace=True)
            self.fno_DF.reset_index(drop=True, inplace=True)
            return True
        except Exception as e:
            print_exception(e)
            return False

    def get_fno_data_for_today(self, symbol):
        return self.get_fno_data_between_dates(symbol, get_current_date())

    def get_fno_data_for_last_n_days(self, symbol, n_days=0):
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days + 5)
        return self.get_fno_data_between_dates(symbol, start_date, end_date)

    def get_last_n_expiry_dates(self, symbol, instrument, n_expiry):
        end_date = get_current_date()
        self.last_stm = select([text('EXPIRY_DT')]).where(and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT <= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(desc(self.fno_table.c.EXPIRY_DT)).limit(n_expiry + 1)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        self.last_DF.sort_values(['EXPIRY_DT'], inplace=True)
        false_expirys = (self.last_DF.EXPIRY_DT - self.last_DF.EXPIRY_DT.shift(1)).dt.days <= 1
        return self.last_DF[~false_expirys]
    
    def get_last_n_expiry_with_starting_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['EX_START'] = df['EXPIRY_DT'].shift(1) + pd.Timedelta('1Day')
        df.dropna(inplace=True)
        df.sort_values(by='EX_START', axis=0, inplace=True)
        return df[['EX_START', 'EXPIRY_DT']]

    def get_last_n_expiry_to_expiry_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['EX_START'] = df['EXPIRY_DT'].shift(1)
        df.dropna(inplace=True)
        df.sort_values(by='EX_START', axis=0, inplace=True)
        return df[['EX_START', 'EXPIRY_DT']]

    @staticmethod
    def get_hlc_difference_with_o(df):
        df['H-O'] = df['HIGH'] - df['OPEN']
        df['L-O'] = df['LOW'] - df['OPEN']
        df['C-O'] = df['CLOSE'] - df['OPEN']
        return df

    @staticmethod
    def get_options_chain(options_df):
        opg = options_df.groupby('OPTION_TYP')
        ce_df = opg.get_group('CE').drop('OPTION_TYP', axis=1)
        pe_df = opg.get_group('PE').drop('OPTION_TYP', axis=1)

        ce_df = OptionsBT.get_hlc_difference_with_o(ce_df)
        pe_df = OptionsBT.get_hlc_difference_with_o(pe_df)

        chain_df = ce_df.merge(pe_df, how='outer',
                               on=['TIMESTAMP', 'SYMBOL', 'EXPIRY_DT', 'STRIKE_PR'],
                               suffixes=['_C', '_P'])
        return chain_df

    @staticmethod
    def calculate_profit(strategy, lot_size, num_lot, brokerage, stop_loss):
        strategy['HL'] = strategy['H-O_C'] + strategy['L-O_P']
        strategy['LH'] = strategy['L-O_C'] + strategy['H-O_P']
        strategy['CC'] = strategy['C-O_C'] + strategy['C-O_P']
        strategy['LOT_SIZE'] = lot_size
        strategy['NUM_LOT'] = num_lot
        qty = lot_size * num_lot
        strategy['QTY'] = lot_size * num_lot
        strategy['BROKERAGE'] = brokerage
        strategy['STOP_LOSS'] = stop_loss
        strategy['HLG_LONG'] = strategy['HL'] * qty
        strategy['LHG_LONG'] = strategy['LH'] * qty
        strategy['CCG_LONG'] = strategy['CC'] * qty
        strategy['HLG_SHORT'] = strategy['HLG_LONG'] * -1
        strategy['LHG_SHORT'] = strategy['LHG_LONG'] * -1
        strategy['CCG_SHORT'] = strategy['CCG_LONG'] * -1
        strategy['NET_LONG'] = strategy['CCG_LONG'] - brokerage
        strategy['NET_SHORT'] = strategy['CCG_SHORT'] - brokerage
        columns = strategy.columns.insert(6, 'DTE')
        strategy['DTE'] = (strategy['EXPIRY_DT'] - strategy['EX_START']).dt.days
        return strategy[columns]

    @staticmethod
    def get_straddle(options_df, spot_df, strike_price=None):
        try:
            options_chain_df = OptionsBT.get_options_chain(options_df)
            straddle_df = spot_df.merge(options_chain_df, how='outer', on=['TIMESTAMP', 'SYMBOL'])
            if strike_price is None:
                straddle_index = straddle_df.groupby('TIMESTAMP').apply(lambda x: (np.abs(x['OPEN'] - x['STRIKE_PR'])).idxmin())
            else:
                straddle_index = straddle_df.groupby('TIMESTAMP').apply(lambda x: (np.abs(strike_price - x['STRIKE_PR'])).idxmin())
            straddle_df = straddle_df.loc[straddle_index].reset_index(drop=True)
            return straddle_df.dropna()
        except Exception as e:
            print_exception(e)
            return None

    def count_losing_streak(self, x):
        if x < 0:
            self.losing_streak_counter += 1
        else:
            self.losing_streak_counter = 0
        return self.losing_streak_counter
    
    def write_to_excel(self, strategy_df, strategy, symbol, n_row, num_lots, lot_size):
        strategy_df['SHORT_LOSE_COUNT'] = strategy_df['NET_SHORT'].apply(self.count_losing_streak)
        strategy_df['LONG_LOSE_COUNT'] = strategy_df['NET_LONG'].apply(self.count_losing_streak)
        columns = strategy_df.columns
        strategy_df.style.set_properties(columns[0:9], **{'background-color': '#ABEBC6'}) \
            .set_properties(columns[9:16], **{'background-color': '#F0FFFF'}) \
            .set_properties(columns[16:23], **{'background-color': '#FFE4E1'}) \
            .set_properties(columns[23:26], **{'background-color': '#E0FFFF'}) \
            .set_properties(columns[27:34], **{'background-color': '#F0FFFF'}) \
            .set_properties(columns[34:40], **{'background-color': '#FFE4E1'}) \
            .set_properties(columns[0:], **{'border-color': 'black'}) \
            .format({'TIMESTAMP': '%Y-%m-%d', 'EXPIRY_DT': '%Y-%m-%d'})\
            .to_excel('{}_{}_{}_{}_{}_{}.xlsx'.
                      format(symbol, strategy, n_row, num_lots, lot_size,
                             get_current_date().strftime('%Y-%m-%d')), engine='openpyxl', index=False)

        return strategy_df

    def index_daily_straddle(self, symbol, lot_size, num_lots, brokerage, stop_loss, n_days=10, strike_price=None):
        if stop_loss > 0:
            stop_loss = stop_loss * -1

        if self.get_fno_data_for_last_n_days(symbol, n_days):
            self.futures_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.IndexFutures]
            self.options_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.IndexOptions]
            self.futures_data.drop(self.futures_data.columns[0:2], axis=1, inplace=True)
            self.options_data.drop(self.options_data.columns[0:2], axis=1, inplace=True)

        self.spot_data = self.get_spot_data_for_last_n_days(symbol, InstrumentType.Index, n_days=n_days)
        straddle_df = OptionsBT.get_straddle(self.options_data, self.spot_data, strike_price)
        straddle_df.rename(columns={'TIMESTAMP':'EX_START'}, inplace=True)
        straddle_df = OptionsBT.calculate_profit(straddle_df, lot_size, num_lots, brokerage, stop_loss)
        self.strategy_df = self.write_to_excel(straddle_df, 'DAILY_STRADDLE', symbol, n_days, num_lots, lot_size)
        return self.strategy_df

    def bank_nifty_daily_straddle_for_last_n_days(self, n_days):
        bn_df = self.index_daily_straddle('BANKNIFTY', lot_size=40, num_lots=2, brokerage=500, stop_loss=-10000, n_days=n_days)
        return bn_df

    @staticmethod
    def get_atm_strike(day, spot_data, options_data, at):
        od_df = options_data[options_data['TIMESTAMP'] >= day]
        sd_df = spot_data[spot_data['TIMESTAMP'] >= day]
        if at == CandleData.OPEN:
            spot = sd_df['OPEN'].iloc[0]
        elif at == CandleData.CLOSE:
            spot = sd_df['CLOSE'].iloc[0]
        atm_i = np.abs(od_df['STRIKE_PR'] - spot).idxmin()
        strike = od_df['STRIKE_PR'].loc[atm_i]
        return strike

    @staticmethod
    def get_strikes_at_price(day, options_data, price, at):
        try:
            cs = options_data[(options_data['TIMESTAMP'] >= day) & (options_data['OPTION_TYP'] == 'CE')]
            ps = options_data[(options_data['TIMESTAMP'] >= day) & (options_data['OPTION_TYP'] == 'PE')]
            if at == CandleData.OPEN:
                ci = np.abs(cs[(cs['TIMESTAMP'] == cs['TIMESTAMP'].iloc[0]) & (cs['EXPIRY_DT'] == cs['EXPIRY_DT'].iloc[0])]['OPEN'] - price).idxmin()
                pi = np.abs(ps[(ps['TIMESTAMP'] == ps['TIMESTAMP'].iloc[0]) & (ps['EXPIRY_DT'] == ps['EXPIRY_DT'].iloc[0])]['OPEN'] - price).idxmin()
            elif at == CandleData.CLOSE:
                ci = np.abs(cs[(cs['TIMESTAMP'] == cs['TIMESTAMP'].iloc[0]) & (cs['EXPIRY_DT'] == cs['EXPIRY_DT'].iloc[0])]['CLOSE'] - price).idxmin()
                pi = np.abs(ps[(ps['TIMESTAMP'] == ps['TIMESTAMP'].iloc[0]) & (ps['EXPIRY_DT'] == ps['EXPIRY_DT'].iloc[0])]['CLOSE'] - price).idxmin()
            call_strike = cs['STRIKE_PR'].loc[ci]
            put_strike = ps['STRIKE_PR'].loc[pi]
            return call_strike, put_strike
        except Exception as ex:
            print_exception(ex)
            return None
    
    def e2e_straddle_builder(self, st, nd, at):
        try:
            print(f'START DATE : {st} | END DATE {nd}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            strike = OptionsBT.get_atm_strike(st, sdi, odi, at)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == strike)]
            dfg = df.groupby('OPTION_TYP')
            cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            days = (cdf.index[-1] - cdf.index[0]).days + 1
            if at == CandleData.OPEN:
                ce_df = OptionsBT.get_hlc_difference_with_o(cdf.resample(f'{days}D').agg(Defaults.CONVERSION))
                pe_df = OptionsBT.get_hlc_difference_with_o(pdf.resample(f'{days}D').agg(Defaults.CONVERSION))
            elif at == CandleData.CLOSE:
                pass
            chain = ce_df.merge(pe_df, how='inner',left_index=True, right_index=True, suffixes=['_C', '_P'])
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            sd['STRIKE'] = strike
            return sd.merge(chain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def e2e_strangle_builder(self, st, nd, price, at):
        try:
            print(f'START DATE : {st} | END DATE {nd}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            cs, ps = OptionsBT.get_strikes_at_price(st, odi, price, at)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd)]
            dfg = df.groupby('OPTION_TYP')
            cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            cdf = cdf[cdf['STRIKE_PR'] == cs]
            pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdf = pdf[pdf['STRIKE_PR'] == ps]
            days = (cdf.index[-1] - cdf.index[0]).days + 1
            if at == CandleData.OPEN:
                ce_df = OptionsBT.get_hlc_difference_with_o(cdf.resample(f'{days}D').agg(Defaults.CONVERSION))
                pe_df = OptionsBT.get_hlc_difference_with_o(pdf.resample(f'{days}D').agg(Defaults.CONVERSION))
            elif at == CandleData.CLOSE:
                pass
            columns = ce_df.columns.insert(0, 'STRIKE')
            ce_df['STRIKE'] = cs
            pe_df['STRIKE'] = ps
            ce_df = ce_df[columns]
            pe_df = pe_df[columns]
            chain = ce_df.merge(pe_df, how='inner',left_index=True, right_index=True, suffixes=['_C', '_P'])
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            rst = sd.merge(chain, how='inner', left_index=True, right_index=True)
            self.results.append(rst)
            return rst
        except Exception as ex:
            print_exception(ex)
            return None

    def prepare_strategy(self, symbol, instrument, n_expiry):
        try:
            symbol = symbol.upper()
            self.expirys = self.get_last_n_expiry_with_starting_dates(symbol, instrument, n_expiry)
            st = self.expirys['EX_START'].iloc[0]
            nd = self.expirys['EXPIRY_DT'].iloc[-1]
            self.spot_data = self.get_spot_data_between_dates(symbol, instrument,  st, nd)

            if self.get_fno_data_between_dates(symbol, st, nd):
                self.futures_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.getfutures(instrument)][Defaults.OPTIONS_COLUMNS]
                self.options_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == instrument][Defaults.OPTIONS_COLUMNS]
            else:
                print('Failed to get fno data')
                return None
        except Exception as ex:
            print_exception(ex)
            return None

    def e2e_straddle(self, symbol, instrument, lot_size, num_lots, stop_loss, brokerage, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            straddle_df = self.expirys.groupby(['EX_START', 'EXPIRY_DT']).apply(lambda x: OptionsBT.e2e_straddle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], self.spot_data, self.options_data, CandleData.OPEN))
            straddle_df = straddle_df.reset_index().drop(['TIMESTAMP'], axis=1)
            straddle_df = OptionsBT.calculate_profit(straddle_df, lot_size, num_lots, brokerage, stop_loss)
            self.strategy_df = self.write_to_excel(straddle_df, 'E2E_STRADDLE', symbol, n_expiry, num_lots, lot_size)
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None
    
    def e2e_strangle(self, symbol, instrument, lot_size, num_lots, stop_loss, brokerage, price=50, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['EX_START', 'EXPIRY_DT'])
            strangle_df = expg.apply(lambda x: self.e2e_strangle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], price, CandleData.OPEN))
            strangle_df = strangle_df.reset_index().drop(['TIMESTAMP'], axis=1)
            strangle_df = OptionsBT.calculate_profit(strangle_df, lot_size, num_lots, brokerage, stop_loss)
            self.strategy_df = self.write_to_excel(strangle_df, f'E2E_STRANGLE{price}', symbol, n_expiry, num_lots, lot_size)
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None

    @classmethod
    def e2e_banknifty_straddle(cls, num_expiry):
        op = cls()
        op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, 10, 10000, 100, n_expiry=num_expiry)
        return op
    
    @classmethod
    def e2e_banknifty_strangle(cls, num_expiry, price):
        op = cls()
        op.e2e_strangle('BANKNIFTY', InstrumentType.IndexOptions, 40, 10, 10000, 100, price=price, n_expiry=num_expiry)
        return op

    @classmethod
    def e2e_nifty_strangle(cls, num_expiry, price):
        op = cls()
        op.e2e_strangle('NIFTY', InstrumentType.IndexOptions, 75, 10, 10000, 100, price=price, n_expiry=num_expiry)
        return op

    @classmethod
    def bank_nifty_daily_straddle_current_day(cls):
        op = cls()
        op.bank_nifty_daily_straddle_for_last_n_days(2)
        return op

    @classmethod
    def bank_nifty_daily_straddle_n_days(cls, ndays=1000):
        op = cls()
        op.bank_nifty_daily_straddle_for_last_n_days(ndays)
        return op

if __name__ == '__main__':
    opn = OptionsBT.e2e_nifty_strangle(2, 100)