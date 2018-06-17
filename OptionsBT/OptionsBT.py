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
        return self.last_DF
    
    def get_last_n_expiry_with_starting_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['EX_START'] = df['EXPIRY_DT'].shift(-1) + pd.Timedelta('1Day')
        df.dropna(inplace=True)
        df.sort_values(by='EX_START', axis=0, inplace=True)
        return df[['EX_START', 'EXPIRY_DT']]

    def get_last_n_expiry_to_expiry_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['EX_START'] = df['EXPIRY_DT'].shift(-1)
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
    def calculate_daily_net(straddle_df, qty, stop_loss, brokerage):
        try:
            straddle_df['HL'] = straddle_df['H-O_C'] + straddle_df['L-O_P']
            straddle_df['LH'] = straddle_df['L-O_C'] + straddle_df['H-O_P']
            straddle_df['CC'] = straddle_df['C-O_C'] + straddle_df['C-O_P']
            straddle_df['QTY'] = qty
            straddle_df['HLG_LONG'] = straddle_df['HL'] * qty
            straddle_df['LHG_LONG'] = straddle_df['LH'] * qty
            straddle_df['CCG_LONG'] = straddle_df['CC'] * qty
            straddle_df['MIN_LONG'] = straddle_df[['HLG_LONG', 'LHG_LONG', 'CCG_LONG']].min(axis=1)
            straddle_df['MAX_LONG'] = straddle_df[['HLG_LONG', 'LHG_LONG', 'CCG_LONG']].max(axis=1)
            straddle_df['MAX_LONG_NET'] = straddle_df['MAX_LONG'] + stop_loss
            straddle_df['NET_LONG'] = np.where(straddle_df['MIN_LONG'] > stop_loss,
                                               np.where(straddle_df['MAX_LONG_NET'] > straddle_df['CCG_LONG'],
                                                        straddle_df['MAX_LONG_NET'], straddle_df['CCG_LONG']),
                                               stop_loss)
            straddle_df['HLG_SHORT'] = straddle_df['HLG_LONG'] * -1
            straddle_df['LHG_SHORT'] = straddle_df['LHG_LONG'] * -1
            straddle_df['CCG_SHORT'] = straddle_df['CCG_LONG'] * -1
            straddle_df['MIN_SHORT'] = straddle_df[['HLG_SHORT', 'LHG_SHORT', 'CCG_SHORT']].min(axis=1)
            straddle_df['MAX_SHORT'] = straddle_df[['HLG_SHORT', 'LHG_SHORT', 'CCG_SHORT']].max(axis=1)
            straddle_df['MAX_SHORT_NET'] = straddle_df['MAX_SHORT'] + stop_loss
            straddle_df['NET_SHORT'] = np.where(straddle_df['MIN_SHORT'] > stop_loss,
                                                np.where(straddle_df['MAX_SHORT_NET'] > straddle_df['CCG_SHORT'],
                                                         straddle_df['MAX_SHORT_NET'], straddle_df['CCG_SHORT']),
                                                stop_loss)
            straddle_df['BROKERAGE'] = brokerage
            straddle_df['GROSS'] = straddle_df['NET_SHORT'] + straddle_df['NET_LONG']
            straddle_df['NET'] = straddle_df['GROSS'] - brokerage

            columns = straddle_df.columns.insert(6, 'DTE')
            straddle_df['DTE'] = (straddle_df['EXPIRY_DT'] - straddle_df['TIMESTAMP']).dt.days
            return straddle_df[columns]
        except Exception as ex:
            print_exception(ex)

    @staticmethod
    def calculate_profit(strategy, lot_size, num_lot, brokerage, position, stop_loss):
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
        strategy['MIN_LONG'] = strategy[['HLG_LONG', 'LHG_LONG', 'CCG_LONG']].min(axis=1)
        strategy['MAX_LONG'] = strategy[['HLG_LONG', 'LHG_LONG', 'CCG_LONG']].max(axis=1)
        strategy['MAX_LONG_NET'] = strategy['MAX_LONG'] + stop_loss
        strategy['NET_LONG'] = np.where(strategy['MIN_LONG'] > stop_loss, np.where(strategy['MAX_LONG_NET'] > strategy['CCG_LONG'], strategy['MAX_LONG_NET'], strategy['CCG_LONG']), stop_loss) - brokerage
        strategy['HLG_SHORT'] = strategy['HLG_LONG'] * -1
        strategy['LHG_SHORT'] = strategy['LHG_LONG'] * -1
        strategy['CCG_SHORT'] = strategy['CCG_LONG'] * -1
        strategy['MIN_SHORT'] = strategy[['HLG_SHORT', 'LHG_SHORT', 'CCG_SHORT']].min(axis=1)
        strategy['MAX_SHORT'] = strategy[['HLG_SHORT', 'LHG_SHORT', 'CCG_SHORT']].max(axis=1)
        strategy['MAX_SHORT_NET'] = strategy['MAX_SHORT'] + stop_loss
        strategy['NET_SHORT'] = np.where(strategy['MIN_SHORT'] > stop_loss, np.where(strategy['MAX_SHORT_NET'] > strategy['CCG_SHORT'], strategy['MAX_SHORT_NET'], strategy['CCG_SHORT']), stop_loss) - brokerage
        columns = strategy.columns.insert(6, 'DTE')
        strategy['DTE'] = (strategy['EXPIRY_DT'] - strategy['TIMESTAMP']).dt.days
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

    def index_daily_straddle(self, symbol, qty, brokerage, stop_loss, n_days=10, strike_price=None):
        if stop_loss > 0:
            stop_loss = stop_loss * -1

        if self.get_fno_data_for_last_n_days(symbol, n_days):
            self.futures_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.IndexFutures]
            self.options_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.IndexOptions]
            self.futures_data.drop(self.futures_data.columns[0:2], axis=1, inplace=True)
            self.options_data.drop(self.options_data.columns[0:2], axis=1, inplace=True)

        self.spot_data = self.get_spot_data_for_last_n_days(symbol, InstrumentType.Index, n_days=n_days)
        straddle_df = OptionsBT.get_straddle(self.options_data, self.spot_data, strike_price)
        self.index_straddle_df = OptionsBT.calculate_daily_net(straddle_df, qty, stop_loss, brokerage)

        self.index_straddle_df['LOSE_COUNT'] = self.index_straddle_df['NET'].apply(self.count_losing_streak)
        columns = self.index_straddle_df.columns
        self.index_straddle_df.style.set_properties(columns[0:9], **{'background-color': '#ABEBC6'}) \
            .set_properties(columns[9:16], **{'background-color': '#F0FFFF'}) \
            .set_properties(columns[16:23], **{'background-color': '#FFE4E1'}) \
            .set_properties(columns[23:26], **{'background-color': '#E0FFFF'}) \
            .set_properties(columns[27:34], **{'background-color': '#F0FFFF'}) \
            .set_properties(columns[34:40], **{'background-color': '#FFE4E1'}) \
            .set_properties(columns[0:], **{'border-color': 'black'}) \
            .format({'TIMESTAMP': '%Y-%m-%d', 'EXPIRY_DT': '%Y-%m-%d'})\
            .to_excel('STRADDLE_{}_{}_{}_{}_{}.xlsx'.
                      format(symbol, n_days, qty, strike_price,
                             get_current_date().strftime('%Y-%m-%d')), engine='openpyxl', index=False)
        return self.index_straddle_df



    def bank_nifty_daily_straddle_for_last_n_days(self, n_days):
        bn_df = self.index_daily_straddle('BANKNIFTY', qty=80, brokerage=500, stop_loss=-2000, n_days=n_days)
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

    @staticmethod
    def e2e_straddle_builder(st, nd, spot_data, options_data, at):
        strike = OptionsBT.get_atm_strike(st, spot_data, options_data, at)
        df = options_data[(options_data['TIMESTAMP'] >= st) & (options_data['EXPIRY_DT'] == nd) & (options_data['STRIKE_PR'] == strike)]
        dfg = df.groupby('OPTION_TYP')
        cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
        pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
        days = (cdf.index[-1] - cdf.index[0]).days + 1
        if at == CandleData.OPEN:
            ce_df = OptionsBT.get_hlc_difference_with_o(cdf.resample(f'{days}D').agg(Defaults.CONVERSION))
            pe_df = OptionsBT.get_hlc_difference_with_o(pdf.resample(f'{days}D').agg(Defaults.CONVERSION))
        elif at == CandleData.CLOSE:
            pass
        chain = ce_df.merge(pe_df, how='outer',left_index=True, right_index=True, suffixes=['_C', '_P'])
        sd = spot_data.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
        sd['STRIKE'] = strike
        return sd.merge(chain, how='inner', left_index=True, right_index=True)

    def e2e_straddle(self, symbol, instrument, lot_size, num_lots, stop_loss, brokerage, initial_capital=100000, n_expiry=10, position_type=PositionType.Long):
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

            result = self.expirys.groupby(['EX_START', 'EXPIRY_DT']).apply(lambda x: OptionsBT.e2e_straddle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], self.spot_data, self.options_data, CandleData.OPEN))

            self.straddle = result

            return result
        except Exception as ex:
            print_exception(ex)
            return None
    
    @classmethod
    def e2e_banknifty_straddle(cls):
        op = cls()
        op.e2e_straddle('BANKNIFTY', InstrumentType.IndexOptions, 40, 10, 10000, 500)
        return op

    @classmethod
    def bank_nifty_daily_straddle_current_day(cls):
        op = cls()
        df = op.bank_nifty_daily_straddle_for_last_n_days(2)
        print(df)

if __name__ == '__main__':
    op = OptionsBT()