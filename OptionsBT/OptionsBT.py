from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from pathlib import Path

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
            self.cum_losing_streak_counter = 0
            self.options_strike_increment = 0
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

    def calculate_profit_2(self, sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage):
        try:
            sdf['LOT_SIZE'] = lot_size
            sdf['NUM_LOTS'] = num_lots
            sdf['TOTAL_LOTS'] = num_lots * 2
            sdf['MARGIN_PER_LOT'] = margin_per_lot
            sdf['MARGIN_REQUIRED'] = margin_per_lot * sdf['TOTAL_LOTS']
            sdf['STOP_LOSS'] = stop_loss
            sdf['STOP_LOSS_TRIGGER'] = stop_loss_threshold
            sdf['NET_PL_LO'] = sdf['PL_LOW'] * num_lots * lot_size
            sdf['NET_PL_HI'] = sdf['PL_HIGH'] * num_lots * lot_size
            sdf['NET_PL_CL'] = sdf['PL_CLOSE'] * num_lots * lot_size
            ####################### STOP LOSS HIT FIRST?*LOW PL INDEX IS < HI PL INDEX & LOW PL IS < STOP LOSS THRESHOLD**OR**HIGHEST PL IS < STOP LOSS & CLOSING PROFIT IS < STOP LOSS ###########
            #################################################################### When highest PL < stop loss means system would not have trailed stop loss and would have hit the default stop loss
            sdf['STOP_LOSS_HIT'] = (sdf['PL_LO_IDX'] < sdf['PL_HI_IDX']) & (sdf['NET_PL_LO'] < (stop_loss_threshold * -1))
            #TRAILING STOP LOSS, CLOSING PROFIT IS LESS THAN HIGH PROFIT - STOP LOSS THRESHOLD
            sdf['SL_HI_GT_CL'] = sdf['NET_PL_CL'] < (sdf['NET_PL_HI'] - stop_loss_threshold)
            #At any given period it would not be possible to exactly stop at the stop loss, but what ever was the loss at end of period
            sdf['GROSS'] = np.where(sdf['STOP_LOSS_HIT'], sdf['NET_PL_LO'], np.where(sdf['SL_HI_GT_CL'], sdf['NET_PL_HI'] - stop_loss, sdf['NET_PL_CL']))
            sdf['BROKERAGE'] = brokerage
            sdf['NET'] = sdf['GROSS'] - brokerage
            sdf['%'] = (sdf['NET']/sdf['MARGIN_REQUIRED'])*100
            sdf['LOSE_COUNT'] = sdf['NET'].apply(self.count_losing_streak)
            sdf['CUM_LOSE_COUNT'] = sdf['NET'].apply(self.count_cumulative_losing_streak)
            sdf['CUM_NET'] = sdf['NET'].cumsum()
            return sdf
        except Exception as e:
            print_exception(e)
            return None

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
    
    def count_cumulative_losing_streak(self, x):
        if x < 0:
            self.cum_losing_streak_counter += 1
        return self.cum_losing_streak_counter

    def set_losing_streak(self, strategy_df):
        strategy_df['SHORT_LOSE_COUNT'] = strategy_df['NET_SHORT'].apply(self.count_losing_streak)
        strategy_df['LONG_LOSE_COUNT'] = strategy_df['NET_LONG'].apply(self.count_losing_streak)
        return strategy_df

    def write_to_excel(self, strategy, symbol, n_row, num_lots, lot_size):
        sdf = self.strategy_df
        columns = sdf.columns
        fname = '{}_{}_{}_{}_{}_{}.xlsx'.format(symbol, strategy, n_row, num_lots, lot_size,datetime.today().strftime('%Y-%m-%d_%H-%M-%S'))
        file_name = Path(__file__).parent.parent.joinpath('Result', fname)
        try:
            sdf.style.set_properties(Defaults.CANDLE_COLUMNS, **{'background-color': '#ABEBC6'}) \
                .set_properties(Defaults.CALL_CANDLE_COLUMNS, **{'background-color': '#F0FFFF'}) \
                .set_properties(Defaults.PUT_CANDLE_COLUMNS, **{'background-color': '#FFE4E1'}) \
                .set_properties(Defaults.PL_CANDLE_COLUMNS, **{'background-color': '#E0FFFF'})\
                .set_properties(Defaults.STOP_LOSS_TRUTH_COLUMNS, **{'background-color': '#F0FFFF'})\
                .set_properties(Defaults.CONSTANT_COLUMNS, **{'background-color': '#FFE4E1'})\
                .set_properties(Defaults.PL_NET_COLUMNS, **{'background-color': '#E0FFFF'})\
                .set_properties(Defaults.NET_COLUMNS, **{'background-color': '#ABEBC6'})\
                .set_properties(columns[0:], **{'border-color': 'black'}) \
                .format({'EX_START': '%Y-%m-%d', 'EXPIRY_DT': '%Y-%m-%d'})\
                .to_excel(file_name, engine='openpyxl', index=False)
        except:
            sdf.to_excel(file_name, engine='openpyxl', index=False)

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
        self.strategy_df = self.set_losing_streak(straddle_df)
        self.write_to_excel('DAILY_STRADDLE', symbol, n_days, num_lots, lot_size)
        return self.strategy_df

    def bank_nifty_daily_straddle_for_last_n_days(self, n_days):
        bn_df = self.index_daily_straddle('BANKNIFTY', lot_size=40, num_lots=2, brokerage=500, stop_loss=-10000, n_days=n_days)
        return bn_df

    def get_options_strike_increment(self):
        try:
            dt = self.options_data['TIMESTAMP'].max()
            ond = self.options_data[self.options_data['TIMESTAMP'] == dt]['STRIKE_PR']
            ond.drop_duplicates(inplace=True)
            ond.sort_values(inplace=True)
            ond = ond - ond.shift(1)
            self.options_strike_increment = ond.dropna().mean()
            return self.options_strike_increment
        except Exception as ex:
            print_exception(ex)
            return None

    def get_atm_strike(self, day, at=CandleData.OPEN):
        od_df = self.options_data[self.options_data['TIMESTAMP'] >= day]
        sd_df = self.spot_data[self.spot_data['TIMESTAMP'] >= day]
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
    
    def e2e_straddle_builder(self, st, nd, position_type):
        try:
            print(f'START DATE : {st} | END DATE {nd}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            strike = self.get_atm_strike(st)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == strike)]
            dfg = df.groupby('OPTION_TYP')
            cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            #Get number of days between start date and end date
            days = (cdf.index[-1] - cdf.index[0]).days + 1
            co = cdf['OPEN'].iloc[0] #Call open price
            po = pdf['OPEN'].iloc[0] #Put open price
            #Create chain
            chain = cdf.merge(pdf, how='inner', left_index=True, right_index=True, suffixes=['_C', '_P'])

            #Calculate end of term position status
            if position_type == PositionType.SHORT:
                chain['DPL'] = (co - chain['CLOSE_C']) + (po - chain['CLOSE_P'])
            elif position_type == PositionType.LONG:
                chain['DPL'] = (chain['CLOSE_C'] - co) + (chain['CLOSE_P'] - po)

            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{days}D').agg(Defaults.STRADDLE_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            sd['STRIKE'] = strike
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None
    
    def e2e_strangle_builder(self, st, nd, price):
        try:
            print(f'START DATE : {st} | END DATE {nd}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            cs, ps = OptionsBT.get_strikes_at_price(st, odi, price, CandleData.OPEN)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == cs)]
            dfg = df.groupby('OPTION_TYP')
            cdfl = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            ccolumns = cdfl.columns.insert(0, 'STRIKE')
            cdfl['STRIKE'] = cs
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == ps)]
            dfg = df.groupby('OPTION_TYP')
            pdfr = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdfr['STRIKE'] = ps
            col = cdfl['OPEN'].iloc[0] #Call open price of short side
            por = pdfr['OPEN'].iloc[0] #Put open price of short side
            #Get number of days between start date and end date
            days = (cdfl.index[-1] - cdfl.index[0]).days + 1
            co = cdfl['OPEN'].iloc[0] #Call open price
            po = pdfr['OPEN'].iloc[0] #Put open price
            #Create chain
            chain = cdfl[ccolumns].merge(pdfr[ccolumns], how='inner', left_index=True, right_index=True, suffixes=['_C', '_P'])
            self.last_DF = chain
            chain['DPL'] = (co - chain['CLOSE_C']) + (po - chain['CLOSE_P'])
            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{days}D').agg(Defaults.STRANGLE_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def iron_butterfly_builder(self, st, nd, position_type):
        try:
            print(f'START DATE : {st} | END DATE {nd}', end='')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            strike = self.get_atm_strike(st)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == strike)]
            dfg = df.groupby('OPTION_TYP')
            cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            #Get number of days between start date and end date
            days = (cdf.index[-1] - cdf.index[0]).days + 1
            co = cdf['OPEN'].iloc[0] #Call open price
            po = pdf['OPEN'].iloc[0] #Put open price
            #Get strikes for short strangle
            #osi = self.get_options_strike_increment() * 3
            #cs, ps = strike + osi, strike - osi
            cs, ps = OptionsBT.get_strikes_at_price(st, odi, 40, CandleData.OPEN)
            #Buy at the money strike straddle
            print(f' | ATM : {strike} | CS : {cs} | PS : {ps}')
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == cs)]
            dfg = df.groupby('OPTION_TYP')
            cdfl = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            ccolumns = cdfl.columns.insert(0, 'STRIKE')
            cdfl['STRIKE'] = cs
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == ps)]
            dfg = df.groupby('OPTION_TYP')
            pdfr = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdfr['STRIKE'] = ps
            col = cdfl['OPEN'].iloc[0] #Call open price of short side
            por = pdfr['OPEN'].iloc[0] #Put open price of short side

            #Create chain
            ln_chain = cdf.merge(pdf, how='inner', left_index=True, right_index=True, suffixes=['_C', '_P'])
            columns = ln_chain.columns.insert(0, 'STRIKE')
            ln_chain['STRIKE'] = strike
            st_chain = cdfl[ccolumns].merge(pdfr[ccolumns], how='inner', left_index=True, right_index=True, suffixes=['_CL', '_PR'])
            chain = ln_chain.merge(st_chain, how='inner', left_index=True, right_index=True, suffixes=['_CL', '_PR'])

            #Calculate end of term position status
            if position_type == PositionType.LONG:
                                #-----------#BUY AT THE MONEY STRIKE------------####-------------SELL OTM STRIKE-----------------------#
                chain['DPL'] = (chain['CLOSE_C'] - co) + (chain['CLOSE_P'] - po) + (col - chain['CLOSE_CL']) + (por - chain['CLOSE_PR'])
            elif position_type == PositionType.SHORT:
                               #-----------#SELL AT THE MONEY STRIKE------------####-------------BUY OTM STRIKE-----------------------#
                chain['DPL'] = (co - chain['CLOSE_C']) + (po - chain['CLOSE_P']) + (chain['CLOSE_CL'] - col) + (chain['CLOSE_PR'] - por)

            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{days}D').agg(Defaults.IRON_BUTTERFLY_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            sd['STRIKE'] = strike
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
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

    def e2e_straddle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10, position_type=PositionType.SHORT):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            sdf = self.expirys.groupby(['EX_START', 'EXPIRY_DT']).apply(lambda x: self.e2e_straddle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], position_type))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{position_type}_E2E_STRADDLE', symbol, n_expiry, num_lots, lot_size)
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None
    
    def e2e_strangle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, price=50, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['EX_START', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.e2e_strangle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], price))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage) 
            self.write_to_excel(f'E2E_STRANGLE{price}', symbol, n_expiry, num_lots, lot_size)
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None
         
    def iron_butterfly(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10, position_type=PositionType.SHORT):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['EX_START', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.iron_butterfly_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], position_type))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{position_type}_E2E_IRB', symbol, n_expiry, num_lots, lot_size)
        except Exception as ex:
            print_exception(ex)
            return None
    
    def calendar_spread_straddle_builder(self, st, nd, nxd):
        try:
            print(f'START DATE : {st} | END DATE {nd}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            strike = self.get_atm_strike(st)
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == strike)]
            dfg = df.groupby('OPTION_TYP')
            cdf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pdf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            #Get number of days between start date and end date
            days = (cdf.index[-1] - cdf.index[0]).days + 1
            co = cdf['OPEN'].iloc[0] #Call open price
            po = pdf['OPEN'].iloc[0] #Put open price
            #Get option data for next expiry same strike
            df = odi[(odi['TIMESTAMP'] >= st) & (odi['TIMESTAMP'] <= nd) & (odi['EXPIRY_DT'] == nxd) & (odi['STRIKE_PR'] == strike)]
            dfg = df.groupby('OPTION_TYP')
            cndf = dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            pndf = dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            con = cndf['OPEN'].iloc[0] #Call open price of long side
            pon = pndf['OPEN'].iloc[0] #Put open price of long side

            #Create chain
            st_chain = cdf.merge(pdf, how='inner', left_index=True, right_index=True, suffixes=['_C', '_P'])
            ln_chain = cndf.merge(pndf, how='inner', left_index=True, right_index=True, suffixes=['_CN', '_PN'])
            chain = st_chain.merge(ln_chain, how='inner', left_index=True, right_index=True, suffixes=['_L', '_R'])
            self.last_DF = chain
            #Calculate end of term position status
                            #-----------#SELL AT THE MONEY STRIKE------------####-------------BUY NEXT EXPIRY ATM STRIKE-----------#
            chain['DPL'] = (co - chain['CLOSE_C']) + (po - chain['CLOSE_P']) + (chain['CLOSE_CN'] - con) + (chain['CLOSE_PN'] - pon)
            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{days}D').agg(Defaults.CALENDAR_SPREAD_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            sd['STRIKE'] = strike
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def calendar_spread_straddle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expsa = self.options_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            self.expirys['NEXPIRY_DT'] = self.expirys['EXPIRY_DT'].apply(lambda x: expsa[expsa[expsa == x].index[0] + 1])
            expg = self.expirys.groupby(['EX_START', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.calendar_spread_straddle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], x['NEXPIRY_DT'].iloc[0]))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'E2E_CALENDAR_SPREAD', symbol, n_expiry, num_lots, lot_size)
        except Exception as ex:
            print_exception(ex)
            return None
    
    def calendar_spread_straddle_only_on_expiry_day(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expsa = self.options_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            self.expirys['EX_START'] = self.expirys['EXPIRY_DT'] - pd.Timedelta('3Day')
            self.expirys['NEXPIRY_DT'] = self.expirys['EXPIRY_DT'].apply(lambda x: expsa[expsa[expsa == x].index[0] + 1])
            expg = self.expirys.groupby(['EX_START', 'EXPIRY_DT', 'NEXPIRY_DT'])
            sdf = expg.apply(lambda x: self.calendar_spread_straddle_builder(x['EX_START'].iloc[0], x['EXPIRY_DT'].iloc[0], x['NEXPIRY_DT'].iloc[0]))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'E2E_CALENDAR_SPREAD', symbol, n_expiry, num_lots, lot_size)
        except Exception as ex:
            print_exception(ex)
            return None

if __name__ == '__main__':
    op = OptionsBT()