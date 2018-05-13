from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime
import pandas as pd
import numpy as np

import dbtables
from util import get_current_date, print_exception, fix_start_and_end_date


class OptionsBT(object):
    def __init__(self):
        try:
            self.db = create_engine('sqlite:///D:/Work/db/bhav.db')
            self.meta_data = MetaData(self.db)
            self.fno_table = Table(dbtables.FNO, self.meta_data, autoload=True)
            self.index_table = Table(dbtables.IINDEX, self.meta_data, autoload=True)
            self.stock_table = Table(dbtables.STOCKS, self.meta_data, autoload=True)
            self.index_data_cache_DF = None
            self.option_data_cache_DF = None
            self.last_DF = None
            self.current_symbol = None
            self.last_stm = None
            self.index_straddle_df = None
            self.losing_streak_counter = 0

        except Exception as e:
            print_exception(e)

    def get_sql_query_statement(self, table, symbol, start_date, end_date=None):
        try:
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            meta = MetaData(self.db)
            dts = Table(table, meta, autoload=True)
            stm = select(['*']).where(
                and_(dts.c.TIMESTAMP >= start_date, dts.c.TIMESTAMP <= end_date, dts.c.SYMBOL == symbol))
            return stm
        except Exception as e:
            print_exception(e)
            return None

    def get_index_data_between_dates(self, symbol, start_date, end_date=None):
        try:
            self.current_symbol = symbol
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            self.last_stm = select(['*']).where(
                and_(self.index_table.c.TIMESTAMP >= start_date, self.index_table.c.TIMESTAMP <= end_date,
                     self.index_table.c.SYMBOL == symbol))
            self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP'])
            self.last_DF.drop([self.last_DF.columns[0]], axis=1, inplace=True)
            self.last_DF.reset_index(drop=True, inplace=True)
            return self.last_DF[self.last_DF.columns[0:6]]
        except Exception as e:
            print_exception(e)
            return None

    def get_index_data_for_today(self, symbol):
        self.current_symbol = symbol
        df = self.get_index_data_between_dates(symbol, get_current_date())
        return df

    def get_index_data_for_last_n_days(self, symbol, n_days=0):
        self.current_symbol = symbol
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days)
        df = self.get_index_data_between_dates(symbol, start_date, end_date)
        return df

    def get_option_data_between_dates(self, symbol, start_date, end_date=None, is_index=True):
        try:
            self.current_symbol = symbol
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            if is_index:
                instrument = 'OPTIDX'
            else:
                instrument = 'OPTSTK'

            self.last_stm = select(['*']).where(
                and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.TIMESTAMP >= start_date,
                     self.fno_table.c.TIMESTAMP <= end_date,
                     self.fno_table.c.SYMBOL == symbol))

            self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP', 'EXPIRY_DT'])
            self.last_DF.sort_values(['OPTION_TYP', 'EXPIRY_DT', 'STRIKE_PR'], inplace=True)
            self.last_DF.drop(self.last_DF.columns[0:2], axis=1, inplace=True)
            self.last_DF.reset_index(drop=True, inplace=True)
            return self.last_DF[self.last_DF.columns[-1:].append(self.last_DF.columns[0:8])]
        except Exception as e:
            print_exception(e)
            return None

    def get_option_data_for_today(self, symbol, is_index=True):
        self.current_symbol = symbol
        df = self.get_option_data_between_dates(symbol, get_current_date(), is_index=is_index)
        return df

    def get_option_data_for_last_n_days(self, symbol, n_days=0, is_index=True):
        self.current_symbol = symbol
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days)
        df = self.get_option_data_between_dates(symbol, start_date, end_date, is_index=is_index)
        return df

    def get_stock_data_between_dates(self, symbol, start_date, end_date=None):
        try:
            self.current_symbol = symbol
            start_date, end_date = fix_start_and_end_date(start_date, end_date)
            self.last_stm = select(['*']).where(
                and_(self.stock_table.c.TIMESTAMP >= start_date, self.stock_table.c.TIMESTAMP <= end_date,
                     self.stock_table.c.SYMBOL == symbol))
            self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['TIMESTAMP'])
            self.last_DF.drop([self.last_DF.columns[0]], axis=1, inplace=True)
            self.last_DF.reset_index(drop=True, inplace=True)
            return self.last_DF[self.last_DF.columns[-2::2].append(self.last_DF.columns[0:6]).drop('SERIES')]
        except Exception as e:
            print_exception(e)
            return None

    def get_stock_data_for_today(self, symbol):
        self.current_symbol = symbol
        df = self.get_stock_data_between_dates(symbol, get_current_date())
        return df

    def get_stock_data_for_last_n_days(self, symbol, n_days=0):
        self.current_symbol = symbol
        end_date = get_current_date()
        start_date = end_date - timedelta(days=n_days)
        df = self.get_stock_data_between_dates(symbol, start_date, end_date)
        return df

    def get_last_n_expiry_dates(self, symbol, n_expiry=10, is_index=True):
        self.current_symbol = symbol
        end_date = get_current_date()

        if is_index:
            instrument = 'OPTIDX'
        else:
            instrument = 'OPTSTK'

        self.last_stm = select([text('EXPIRY_DT')]).where(
            and_(self.fno_table.c.INSTRUMENT == instrument, self.fno_table.c.EXPIRY_DT <= end_date,
                 self.fno_table.c.SYMBOL == symbol)).distinct().\
            order_by(desc(self.fno_table.c.EXPIRY_DT)).limit(n_expiry + 1)
        self.last_DF = pd.read_sql_query(self.last_stm, con=self.db, parse_dates=['EXPIRY_DT'])
        return self.last_DF

    def count_losing_streak(self, x):
        if x < 0:
            self.losing_streak_counter += 1
        else:
            self.losing_streak_counter = 0
        return self.losing_streak_counter

    def get_e2e_straddle_normal(self, symbol, start_date, end_date, is_index=True):
        try:
            print('START {0} : END {1}'.format(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
            if self.index_data_cache_DF is None:
                spot_df = self.get_index_data_between_dates(symbol, start_date, end_date)
            else:
                spot_df = self.index_data_cache_DF[(self.index_data_cache_DF['TIMESTAMP'] >= start_date) &
                                                   (self.index_data_cache_DF['TIMESTAMP'] == end_date)]

            start_date = spot_df['TIMESTAMP'].iloc[0]

            if self.option_data_cache_DF is None:
                opti_df = self.get_option_data_between_dates(symbol, start_date, end_date, is_index)
            else:
                opti_df = self.option_data_cache_DF[(self.option_data_cache_DF['TIMESTAMP'] >= start_date) &
                                                    (self.option_data_cache_DF['TIMESTAMP'] == end_date)]

            opfl_df = opti_df[(opti_df['TIMESTAMP'] >= start_date) & (opti_df['EXPIRY_DT'] == end_date)]
            spot = spot_df['OPEN'].iloc[0]
            strike_i = np.abs(opfl_df['STRIKE_PR']-spot).idxmin()
            strike = opti_df['STRIKE_PR'].loc[strike_i]
            options_df = opti_df[(opti_df['TIMESTAMP'] >= start_date) & (opti_df['EXPIRY_DT'] == end_date) & (
                        opti_df['STRIKE_PR'] == strike)]

            opg = options_df.groupby('OPTION_TYP')
            self.ce_df = opg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            self.pe_df = opg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            days = (end_date - start_date).days
            # dayss = '{}D'.format(days)
            # dayss = '7D'
            conversion = {'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last'}
            # cer = self.ce_df.resample(dayss).agg(conversion)
            # per = self.pe_df.resample(dayss).agg(conversion)

            ce_df = OptionsBT.get_hlc_difference_with_o(self.ce_df)
            pe_df = OptionsBT.get_hlc_difference_with_o(self.pe_df)

            self.chain_df = ce_df.merge(pe_df, how='outer',
                                   left_index=True, right_index=True,
                                   suffixes=['_C', '_P'])
            spr = spot_df.set_index('TIMESTAMP').resample('7D').agg(conversion)
            spr['STRIKE'] = strike
            self.straddle_df = spr.merge(self.chain_df, how='outer', left_index=True, right_index=True)
            return self.straddle_df
        except Exception as ex:
            print_exception(ex)
            return None

    def expiry_to_expiry_straddle(self, symbol, qty, stop_loss, brokerage, initial_capital=100000, n_expiry=10,
                                  is_index=True):
        try:
            self.current_symbol = symbol.upper()
            expirys = self.get_last_n_expiry_dates(self.current_symbol, n_expiry, is_index)
            expirys['EX_START'] = expirys['EXPIRY_DT'].shift(-1)
            expirys.dropna(inplace=True)
            st = expirys['EX_START'].iloc[0]
            nd = expirys['EXPIRY_DT'].iloc[-1]
            self.index_data_cache_DF = self.get_index_data_between_dates(self.current_symbol, st, nd)
            self.option_data_cache_DF = self.get_option_data_between_dates(self.current_symbol, st, nd)
            exps = expirys.groupby(['EX_START', 'EXPIRY_DT']).\
                apply(lambda x: self.get_e2e_straddle_normal(self.current_symbol,
                                                             x['EX_START'].iloc[0],
                                                             x['EXPIRY_DT'].iloc[0]))
            straddle_df = exps.reset_index(level=[1, 2])
            straddle_df['HL'] = straddle_df['H-O_C'] + straddle_df['L-O_P']
            straddle_df['LH'] = straddle_df['L-O_C'] + straddle_df['H-O_P']
            straddle_df['CC'] = straddle_df['C-O_C'] + straddle_df['C-O_P']
            straddle_df['QTY'] = qty
            straddle_df['HLG_L'] = straddle_df['HL'] * qty
            straddle_df['LHG_L'] = straddle_df['LH'] * qty
            straddle_df['CCG_L'] = straddle_df['CC'] * qty
            straddle_df['MIN_L'] = straddle_df[['HLG_L', 'LHG_L']].min(axis=1)
            straddle_df['MAX_L'] = straddle_df[['HLG_L', 'LHG_L']].max(axis=1)
            straddle_df['MAX_L_NET'] = straddle_df['MAX_L'] + stop_loss
            straddle_df['GROSS'] = np.where(straddle_df['MIN_L'] > stop_loss,
                                            np.where(straddle_df['MAX_L_NET'] > straddle_df['CCG_L'],
                                                     straddle_df['MAX_L_NET'], straddle_df['CCG_L']),
                                            stop_loss)
            straddle_df['BROKERAGE'] = brokerage
            straddle_df['NET'] = straddle_df['GROSS'] - brokerage
            straddle_df['CAPITAL_REQ'] = qty * (straddle_df['OPEN_C'] + straddle_df['OPEN_P'])
            straddle_df['CAPITAL_INI'] = initial_capital
            straddle_df['CAPITAL_NET'] = initial_capital + straddle_df['NET'].cumsum()
            straddle_df['CAPITAL_INI'] = straddle_df['CAPITAL_NET'].shift(1)
            straddle_df['CAPITAL_INI'].iat[0] = initial_capital
            straddle_df['LOSE_COUNT'] = straddle_df['NET'].apply(self.count_losing_streak)
            return straddle_df
        except Exception as ex:
            print_exception(ex)

    def get_e2e_straddle_by_close(self, symbol, entry_date, start_date, end_date, is_index=True):
        try:
            print('ENTER_ON {} : START {} : END {}'.format(entry_date.strftime('%Y-%m-%d'),
                                                           start_date.strftime('%Y-%m-%d'),
                                                           end_date.strftime('%Y-%m-%d')))
            if self.index_data_cache_DF is None:
                spot_df = self.get_index_data_between_dates(symbol, entry_date, end_date)
            else:
                spot_df = self.index_data_cache_DF[(self.index_data_cache_DF['TIMESTAMP'] >= entry_date) &
                                                   (self.index_data_cache_DF['TIMESTAMP'] <= end_date)]

            start_date = spot_df['TIMESTAMP'].iloc[0]
            spot = self.index_data_cache_DF[(self.index_data_cache_DF['TIMESTAMP'] >= start_date)]['CLOSE'].iloc[0]

            if self.option_data_cache_DF is None:
                opti_df = self.get_option_data_between_dates(symbol, entry_date, end_date, is_index)
            else:
                opti_df = self.option_data_cache_DF[(self.option_data_cache_DF['TIMESTAMP'] >= entry_date) &
                                                    (self.option_data_cache_DF['TIMESTAMP'] <= end_date)]

            opfl_df = opti_df[(opti_df['TIMESTAMP'] >= start_date) & (opti_df['EXPIRY_DT'] == end_date)]

            strike_i = np.abs(opfl_df['STRIKE_PR']-spot).idxmin()
            strike = opti_df['STRIKE_PR'].loc[strike_i]
            options_df = opti_df[(opti_df['TIMESTAMP'] >= start_date) & (opti_df['EXPIRY_DT'] == end_date) & (
                        opti_df['STRIKE_PR'] == strike)]

            opg = options_df.groupby('OPTION_TYP')
            self.ce_df = opg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            self.pe_df = opg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            days = (end_date - start_date).days + 1
            dayss = '{}D'.format(days)
            print(dayss)
            # dayss = '7D'
            conversion = {'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last'}
            cer = self.ce_df.resample(dayss).agg(conversion)
            per = self.pe_df.resample(dayss).agg(conversion)

            ce_df = OptionsBT.get_hlc_difference_with_o(cer)
            pe_df = OptionsBT.get_hlc_difference_with_o(per)

            chain_df = ce_df.merge(pe_df, how='outer',
                                   left_index=True, right_index=True,
                                   suffixes=['_C', '_P'])
            spr = spot_df.set_index('TIMESTAMP').resample(dayss).agg(conversion)
            spr['STRIKE'] = strike
            straddle_df = spr.merge(chain_df, how='outer', left_index=True, right_index=True)
            return straddle_df
        except Exception as ex:
            print_exception(ex)
            return None

    def expiry_to_expiry_straddle_by_close(self, symbol, qty, stop_loss, brokerage, initial_capital=100000, n_expiry=10,
                                           is_index=True):
        try:
            self.current_symbol = symbol.upper()
            expirys = self.get_last_n_expiry_dates(self.current_symbol, n_expiry, is_index)
            expirys['EX_START'] = expirys['EXPIRY_DT'].shift(-1) + pd.Timedelta(days=1)
            expirys['ENTER_ON'] = expirys['EXPIRY_DT'].shift(-1)
            expirys.dropna(inplace=True)

            st = expirys['ENTER_ON'].iloc[0]
            nd = expirys['EXPIRY_DT'].iloc[-1]
            self.index_data_cache_DF = self.get_index_data_between_dates(self.current_symbol, st, nd)
            self.option_data_cache_DF = self.get_option_data_between_dates(self.current_symbol, st, nd)

            exps = expirys.groupby(['EX_START', 'EXPIRY_DT']).apply(lambda x:
                                                                    self.get_e2e_straddle_by_close(self.current_symbol,
                                                                                                   x['ENTER_ON'].iloc[0],
                                                                                                   x['EX_START'].iloc[0],
                                                                                                   x['EXPIRY_DT'].iloc[0]))
            straddle_df = exps.reset_index(level=[1, 2])
            straddle_df['HL'] = straddle_df['H-O_C'] + straddle_df['L-O_P']
            straddle_df['LH'] = straddle_df['L-O_C'] + straddle_df['H-O_P']
            straddle_df['CC'] = straddle_df['C-O_C'] + straddle_df['C-O_P']
            straddle_df['QTY'] = qty
            straddle_df['HLG_L'] = straddle_df['HL'] * qty
            straddle_df['LHG_L'] = straddle_df['LH'] * qty
            straddle_df['CCG_L'] = straddle_df['CC'] * qty
            straddle_df['MIN_L'] = straddle_df[['HLG_L', 'LHG_L']].min(axis=1)
            straddle_df['MAX_L'] = straddle_df[['HLG_L', 'LHG_L']].max(axis=1)
            straddle_df['MAX_L_NET'] = straddle_df['MAX_L'] + stop_loss
            straddle_df['GROSS'] = np.where(straddle_df['MIN_L'] > stop_loss,
                                            np.where(straddle_df['MAX_L_NET'] > straddle_df['CCG_L'],
                                                     straddle_df['MAX_L_NET'], straddle_df['CCG_L']), stop_loss)
            straddle_df['BROKERAGE'] = brokerage
            straddle_df['NET'] = straddle_df['GROSS'] - brokerage
            straddle_df['CAPITAL_REQ'] = qty * (straddle_df['OPEN_C'] + straddle_df['OPEN_P'])
            straddle_df['CAPITAL_INI'] = initial_capital
            straddle_df['CAPITAL_NET'] = initial_capital + straddle_df['NET'].cumsum()
            straddle_df['CAPITAL_INI'] = straddle_df['CAPITAL_NET'].shift(1)
            straddle_df['CAPITAL_INI'].iat[0] = initial_capital
            straddle_df['LOSE_COUNT'] = straddle_df['NET'].apply(self.count_losing_streak)
            return straddle_df
        except Exception as ex:
            print_exception(ex)

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
    def calculate_net(straddle_df, qty, stop_loss, brokerage):
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
    def get_straddle(options_df, spot_df, qty, brokerage, stop_loss, strike_price=None):
        try:
            # If the given stop loss is greater than 0 flip the sign
            if stop_loss > 0:
                print('Changing given stop loss to negative ', stop_loss)
                stop_loss = stop_loss * -1
                print('Given stop loss changed to negative ', stop_loss)

            options_chain_df = OptionsBT.get_options_chain(options_df)

            straddle_df = spot_df.merge(options_chain_df, how='outer', on=['TIMESTAMP', 'SYMBOL'])
            if strike_price is None:
                straddle_index = straddle_df.groupby('TIMESTAMP').apply(
                    lambda x: (np.abs(x['OPEN'] - x['STRIKE_PR'])).idxmin())
            else:
                straddle_index = straddle_df.groupby('TIMESTAMP').apply(
                    lambda x: (np.abs(strike_price - x['STRIKE_PR'])).idxmin())

            straddle_df = straddle_df.loc[straddle_index].reset_index(drop=True)
            return OptionsBT.calculate_net(straddle_df, qty, stop_loss, brokerage)
        except Exception as e:
            print_exception(e)
            return None

    def index_daily_straddle(self, symbol, qty, brokerage, stop_loss, n_days=10, strike_price=None):
        options_df = self.get_option_data_for_last_n_days(symbol, n_days=n_days, is_index=True)
        spot_df = self.get_index_data_for_last_n_days(symbol, n_days=n_days)
        self.index_straddle_df = self.get_straddle(options_df, spot_df, qty,
                                                   brokerage, stop_loss, strike_price=strike_price)

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

    def bank_nifty_straddle_for_one_year(self):
        bn_df = self.bank_nifty_straddle_for_last_n_days(n_days=370)
        return bn_df

    def bank_nifty_straddle_for_last_n_days(self, n_days):
        bn_df = self.index_daily_straddle('BANKNIFTY', qty=80, brokerage=500, stop_loss=-2000, n_days=n_days)
        return bn_df

    def daily_nifty_bank_nifty(self):
        ndf = self.index_daily_straddle('NIFTY', qty=150, brokerage=500, stop_loss=-2000, n_days=2)
        print(ndf)
        bndf = self.index_daily_straddle('BANKNIFTY', qty=80, brokerage=500, stop_loss=-2000, n_days=2)
        print(bndf)

    def get_straddle_at_strike(self, symbol, strike_price, n_days=2):
        df = self.index_daily_straddle(symbol, qty=80, brokerage=500, stop_loss=-2000,
                                       n_days=n_days, strike_price=strike_price)
        print(df)

    def daily_bank_nifty_at_strike(self, strike_price):
        bndf = self.index_daily_straddle('BANKNIFTY', qty=80, brokerage=500, stop_loss=-2000,
                                         n_days=2, strike_price=strike_price)
        print(bndf)

    def bank_nifty_expiry_to_expiry(self, n_expiry=10):
        ebnf = self.expiry_to_expiry_straddle('BANKNIFTY', 40*4, -10000, 500, n_expiry=n_expiry, initial_capital=100000)
        dts = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
        ebnf.to_excel(f'EXPIRY_TO_EXPIRY_bank_nifty_{n_expiry}_{dts}.xlsx')
        self.last_DF = ebnf
        return ebnf

    def nifty_expiry_to_expiry(self, n_expiry=10):
        ebnf = self.expiry_to_expiry_straddle('NIFTY', 75*4, -15000, 500, n_expiry=n_expiry)
        dts = datetime.now().strftime('%Y_%m_%d-%H-%M-%S')
        ebnf.to_excel(f'EXPIRY_TO_EXPIRY_nifty_{n_expiry}_{dts}.xlsx')
        self.last_DF = ebnf
        return ebnf


if __name__ == '__main__':
    op = OptionsBT()
