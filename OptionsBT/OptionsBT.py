from sqlalchemy import create_engine, Table, MetaData, and_, text
from sqlalchemy.sql import select, desc, asc
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns

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
        df['START_DT'] = df['EXPIRY_DT'].shift(1) + pd.Timedelta('1Day')
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]

    def get_last_n_expiry_to_expiry_dates(self, symbol, instrument, n_expiry):
        df = self.get_last_n_expiry_dates(symbol, instrument, n_expiry)
        df['START_DT'] = df['EXPIRY_DT'].shift(1)
        df.dropna(inplace=True)
        df.sort_values(by='START_DT', axis=0, inplace=True)
        return df[['START_DT', 'EXPIRY_DT']]

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
        strategy['DTE'] = (strategy['EXPIRY_DT'] - strategy['START_DT']).dt.days
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
    
    def calculate_profit_double_ratio(self, sdf, lot_size, num_lots, margin_per_lot, brokerage):
        try:
            sdf['LOT_SIZE'] = lot_size
            sdf['NUM_LOTS'] = num_lots
            sdf['TOTAL_LOTS'] = num_lots * 2
            sdf['MARGIN_PER_LOT'] = margin_per_lot
            sdf['MARGIN_REQUIRED'] = margin_per_lot * sdf['TOTAL_LOTS']
            sdf['GROSS'] = sdf['PL_CLOSE']
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

    @staticmethod
    def get_end_of_month(x, eoml):
        try:
            wom =  (x['EXPIRY_DT'].day - 1)// 7+1
            st = x['EXPIRY_DT']
            me = eoml.searchsorted(st)
            if wom >= 4:
                return eoml.iloc[me[0] +1]
            else:
                return eoml.iloc[me[0]]
        except:
            return None

    @staticmethod
    def color_negative_red(val):
        """
        Takes a scalar and returns a string with
        the css property `'color: red'` for negative
        strings, black otherwise.
        """
        if type(val) == float:
            color = 'red' if val < 0 else 'black'
            return 'color: %s' % color
        else:
            return 'color: black'

    def write_to_excel(self, strategy):
        sdf = self.strategy_df
        columns = sdf.columns
        fname = f'{strategy}_{datetime.today():%Y-%m-%d_%H-%M-%S}.xlsx'
        file_name = Path(__file__).parent.parent.joinpath('Result', fname)
        try:
            styl = sdf.style

            #Get price columns
            pcols = [columns.get_loc(x) for x in columns if 'OPEN' in x] 
            pcols1 = pcols[::2]
            pcols2 = pcols[1::2]
            for x in pcols1:
                styl = styl.set_properties(columns[range(x, x+4)], **{'background-color': '#ffc0cb'})
            for x in pcols2:
                styl = styl.set_properties(columns[range(x, x+4)], **{'background-color': '#c6e2ff'})

            #Get date columns
            dcols = [columns.get_loc(x) for x in columns if '_DT' in x]
            platelets = sns.color_palette("Set3", len(dcols)).as_hex()
            for i, x in enumerate(dcols):
                styl = styl.set_properties(columns[x], **{'background-color': platelets[i]})              
            
            #Net columns
            cols = [columns.get_loc(x) for x in columns if 'NET' in x]
            plts = sns.color_palette("Paired", len(cols)).as_hex()
            for i, x in enumerate(cols):
                styl = styl.set_properties(columns[x], **{'background-color': plts[i]})

            #Count columns
            cols = [columns.get_loc(x) for x in columns if 'COUNT' in x]
            plts = sns.color_palette("husl", len(cols)).as_hex()
            for i, x in enumerate(cols):
                styl = styl.set_properties(columns[x], **{'background-color': plts[i]})
            
            #LOT columns
            cols = [columns.get_loc(x) for x in columns if 'LOT' in x]
            plts = sns.color_palette("hls", len(cols)).as_hex()
            for i, x in enumerate(cols):
                styl = styl.set_properties(columns[x], **{'background-color': plts[i]})

            #Set negative values to red
            styl.applymap(OptionsBT.color_negative_red)

            #Set date format not working
            #alphas = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
            #for x in dcols:
            #    if x < len(alphas):
            #        styl = styl.format({alphas[x]: '%Y-%m-%d'})

            styl.set_properties(columns[0:], **{'border-color': 'black'}) \
                .to_excel(file_name, engine='openpyxl', index=False)
        except Exception as ex:
            print_exception(ex)
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
        straddle_df.rename(columns={'TIMESTAMP':'START_DT'}, inplace=True)
        straddle_df = OptionsBT.calculate_profit(straddle_df, lot_size, num_lots, brokerage, stop_loss)
        self.strategy_df = self.set_losing_streak(straddle_df)
        self.write_to_excel(f'{symbol}_DAILY_STRADDLE_{num_lots}_{n_days}')
        return self.strategy_df

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
    
    def get_atm_strike_using_options_data(self, st, expiry, at=CandleData.OPEN):
        try:
            od_df = self.options_data[(self.options_data['TIMESTAMP'] >= st) & (self.options_data['EXPIRY_DT'] == expiry)]

            if at == CandleData.CLOSE:
                od_df = od_df[od_df['CLOSE'] > 0]
            elif at == CandleData.OPEN:
                od_df = od_df[od_df['OPEN'] > 0]
            
            opg = od_df.groupby('OPTION_TYP')
            opc = opg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            opc = opc[opc.index == opc.index[0]]
            opp = opg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            opp = opp[opp.index == opp.index[0]]
            opm = opc.merge(opp, how='inner', left_index=True, on=['STRIKE_PR'], suffixes=['_C', '_P'])
            #Starting day may not be a trading day, hence do this.
            
            if at == CandleData.CLOSE:
                idx = (opm['CLOSE_C'] - opm['CLOSE_P']).abs().reset_index(drop=True).idxmin()
            elif at == CandleData.OPEN:
                idx = (opm['OPEN_C'] - opm['OPEN_P']).abs().reset_index(drop=True).idxmin()

            return opm['STRIKE_PR'].iloc[idx]
        except Exception as ex:
            print(f'EXCEPTION AT | DAY : {st:%Y-%m-%d} - EXPIRY : {expiry:%Y-%m-%d}')
            print_exception(ex)
            return None
    
    def get_max_oi_pcr(self, od_df):
        try:
            opg = od_df.groupby('OPTION_TYP')
            opc = opg.get_group('CE').drop('OPTION_TYP', axis=1)
            coi = opc['OPEN_INT'].max()
            cs = opc[opc['OPEN_INT'] == coi]['STRIKE_PR'].iloc[0]
            opp = opg.get_group('PE').drop('OPTION_TYP', axis=1)
            poi = opp['OPEN_INT'].max()
            ps = opp[opp['OPEN_INT'] == poi]['STRIKE_PR'].iloc[0]
            tcoi = opc['OPEN_INT'].sum()
            tpoi = opp['OPEN_INT'].sum()
            pcr = tpoi/tcoi
            df = pd.DataFrame([[coi, cs, poi, ps, tcoi, tpoi, pcr]], columns=['COI', 'CS', 'POI', 'PS', 'TCOI', 'TPOI', 'PCR'])
            return df
        except Exception as ex:
            print(f'Calculating maxi oi and pcr')
            print_exception(ex)
            return None
    
    def prepare_strategy(self, symbol, instrument, n_expiry):
        try:
            symbol = symbol.upper()
            self.expirys = self.get_last_n_expiry_with_starting_dates(symbol, instrument, n_expiry)
            st = self.expirys['START_DT'].iloc[0]
            nd = self.expirys['EXPIRY_DT'].iloc[-1]
            self.spot_data = self.get_spot_data_between_dates(symbol, instrument,  st, nd)

            if self.get_fno_data_between_dates(symbol, st, nd):
                self.futures_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == InstrumentType.getfutures(instrument)][Defaults.OPTIONS_COLUMNS]
                self.options_data = self.fno_DF[self.fno_DF['INSTRUMENT'] == instrument][Defaults.OPTIONS_COLUMNS]
                opcr = self.options_data.groupby(['EXPIRY_DT', 'TIMESTAMP']).apply(lambda x: self.get_max_oi_pcr(x))
                self.oi_pcr = opcr.reset_index().drop(columns=['level_2'])
            else:
                print('Failed to get fno data')
                return None
        except Exception as ex:
            print_exception(ex)
            return None

    def get_atm_strike_from_futures(self, day):
        try:
            odf = self.options_data[self.options_data['TIMESTAMP'] >= day]
            fdf = self.futures_data[self.futures_data['TIMESTAMP'] >= day]
            ce = fdf['EXPIRY_DT'].drop_duplicates().iloc[0]
            fdi = fdf[fdf['EXPIRY_DT'] == ce]
            future = fdi['OPEN'].iloc[0]
            atm_i = np.abs(odf['STRIKE_PR'] - future).idxmin()
            strike = odf['STRIKE_PR'].loc[atm_i]
            return strike, fdi
        except Exception as ex:
            print_exception(ex)
            return None
    
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
            #strike = self.get_atm_strike(st)
            strike = self.get_atm_strike_using_options_data(st, nd)
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
            sd['PCR'] = self.oi_pcr[(self.oi_pcr['TIMESTAMP'] == st) & (self.oi_pcr['EXPIRY_DT'] == nd)]['PCR'].iloc[0]
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
            sd['PCR'] = self.oi_pcr[(self.oi_pcr['TIMESTAMP'] == st) & (self.oi_pcr['EXPIRY_DT'] == nd)]['PCR'].iloc[0]
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def iron_butterfly_builder(self, st, nd, position_type):
        try:
            print(f'START DATE : {st} | END DATE {nd}', end='')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            #strike = self.get_atm_strike(st)
            strike = self.get_atm_strike_using_options_data(st, nd)
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
            sd['PCR'] = self.oi_pcr[(self.oi_pcr['TIMESTAMP'] == st) & (self.oi_pcr['EXPIRY_DT'] == nd)]['PCR'].iloc[0]
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def e2e_straddle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10, position_type=PositionType.SHORT):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            sdf = self.expirys.groupby(['START_DT', 'EXPIRY_DT']).apply(lambda x: self.e2e_straddle_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], position_type))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{symbol}_{position_type}_E2E_STRADDLE_{num_lots}_{n_expiry}')
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None
    
    def e2e_strangle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, price=50, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.e2e_strangle_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], price))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage) 
            self.write_to_excel(f'{symbol}_E2E_STRANGLE_{price}_{num_lots}_{n_expiry}')
            return self.strategy_df
        except Exception as ex:
            print_exception(ex)
            return None
         
    def iron_butterfly(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10, position_type=PositionType.SHORT):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.iron_butterfly_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], position_type))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{symbol}_{position_type}_E2E_IRB_{num_lots}_{n_expiry}')
        except Exception as ex:
            print_exception(ex)
            return None
    
    def calendar_spread_straddle_builder(self, st, nd, nxd):
        try:
            print(f'START DATE : {st:%Y-%m-%d} | END DATE : {nd:%Y-%m-%d} | LONG EXPIRY : {nxd:%Y-%m-%d}')
            odi = self.options_data
            strike = self.get_atm_strike_using_options_data(st, nd)
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
            if con == 0 or pon == 0:
                chain['DPL'] = (co - chain['CLOSE_C']) + (po - chain['CLOSE_P'])
            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{days}D').agg(Defaults.CALENDAR_SPREAD_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            sd = sdi.set_index('TIMESTAMP').resample(f'{days}D').agg(Defaults.CONVERSION)
            sd['STRIKE'] = strike
            sd['PCR'] = self.oi_pcr[(self.oi_pcr['TIMESTAMP'] == st) & (self.oi_pcr['EXPIRY_DT'] == nd)]['PCR'].iloc[0]
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def calendar_spread_straddle(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            #Get all the expiry dates
            expsa = self.options_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            #Get monthly expiry dates
            mexp = self.futures_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            self.expirys['NEXPIRY_DT'] = self.expirys.apply(lambda x: OptionsBT.get_end_of_month(x, mexp), axis=1)
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT', 'NEXPIRY_DT'])
            sdf = expg.apply(lambda x: self.calendar_spread_straddle_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], x['NEXPIRY_DT'].iloc[0]))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{symbol}_E2E_CALENDAR_SPREAD_{num_lots}_{n_expiry}')
        except Exception as ex:
            print_exception(ex)
            return None
    
    def calendar_spread_straddle_only_on_expiry_day(self, symbol, instrument, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage, n_expiry=10):
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expsa = self.options_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            #Get monthly expiry dates
            mexp = self.futures_data['EXPIRY_DT'].drop_duplicates().reset_index(drop=True)
            self.expirys['NEXPIRY_DT'] = self.expirys.apply(lambda x: OptionsBT.get_end_of_month(x, mexp), axis=1)
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT', 'NEXPIRY_DT'])
            sdf = expg.apply(lambda x: self.calendar_spread_straddle_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], x['NEXPIRY_DT'].iloc[0]))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_2(sdf, lot_size, num_lots, margin_per_lot, stop_loss, stop_loss_threshold, brokerage)
            self.write_to_excel(f'{symbol}_E2E_CALENDAR_SPREAD_EXPIRY_DAY_ONLY_{num_lots}_{n_expiry}')
        except Exception as ex:
            print_exception(ex)
            return None

    def double_ratio_spread_builder(self, st, nd, lot_size, r1=2, r2=3):
        """ double ratio spread builder
        ratio - ratio at which the spread needs to be constructed 2:3:1 is standard, 3:4:1, 4:5:1, 3:5:2 are all valid 
        Parameters:
            st - start date
            nd - end date
            lot_size - size of the lot
            r1 - first part of ratio write - long leg - shall be starting from 2
            r2 - second part of ratio write - short leg - shall be r1 + at least 1
        """
        try:
            #atm = self.get_atm_strike(st)
            atm = self.get_atm_strike_using_options_data(st, nd)
            #Strike increment
            si = self.get_options_strike_increment()
            #Short Call Strike
            scs = atm + si
            #Short Put Strike
            sps = atm - si
            #Long Call Strike
            lcs = scs + si
            #Long Put Strike
            lps = sps - si

            r3 = r2 - r1 #Ratio 3rd part of it

            print(f'Start : {st:%Y-%m-%d} | End : {nd:%Y-%m-%d} | ATM : {atm} | SCS : {scs} | LCS : {lcs} | SPS : {sps} | LPS : {lps}')
            odi = self.options_data
            sdi = self.spot_data[(self.spot_data['TIMESTAMP'] >= st) & (self.spot_data['TIMESTAMP'] <= nd)]
            #Build ATM
            atm_df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == atm)]
            atm_dfg = atm_df.groupby('OPTION_TYP')
            atm_c = atm_dfg.get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            atm_p = atm_dfg.get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            atm_d = atm_c.merge(atm_p, how='inner', left_index=True, right_index=True, suffixes=['_CA', '_PA'])
            acl = atm_d.columns.insert(0, 'STRIKE_ATM')
            atm_d['STRIKE_ATM'] = atm
            atm_d = atm_d[acl]
            #Build short
            scs_df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == scs)]
            scs_df = scs_df.groupby('OPTION_TYP').get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            acl = scs_df.columns.insert(0, 'STRIKE')
            scs_df['STRIKE'] = scs
            scs_df = scs_df[acl]
            sps_df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == sps)]
            sps_df = sps_df.groupby('OPTION_TYP').get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            sps_df['STRIKE'] = sps
            sps_df = sps_df[acl]
            short_df = scs_df.merge(sps_df, how='inner', left_index=True, right_index=True, suffixes=['_CS', '_PS'])
            #Build Long
            lcs_df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == lcs)]
            lcs_df = lcs_df.groupby('OPTION_TYP').get_group('CE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            acl = lcs_df.columns.insert(0, 'STRIKE')
            lcs_df['STRIKE'] = lcs
            lcs_df = lcs_df[acl]
            lps_df = odi[(odi['TIMESTAMP'] >= st) & (odi['EXPIRY_DT'] == nd) & (odi['STRIKE_PR'] == lps)]
            lps_df = lps_df.groupby('OPTION_TYP').get_group('PE').drop('OPTION_TYP', axis=1).set_index('TIMESTAMP')
            lps_df['STRIKE'] = lps
            lps_df = lps_df[acl]
            long_df = lcs_df.merge(lps_df, how='inner', left_index=True, right_index=True, suffixes=['_CL', '_PL'])
            chain = pd.concat([atm_d, short_df, long_df], axis=1)[Defaults.DOUBLE_RATIO_COLUMNS]
            self.last_DF = chain
            # Get all the required open prices
            aco = atm_c['OPEN'].iloc[0]
            apo = atm_p['OPEN'].iloc[0]
            sco = scs_df['OPEN'].iloc[0]
            spo = sps_df['OPEN'].iloc[0]
            lco = lcs_df['OPEN'].iloc[0]
            lpo = lps_df['OPEN'].iloc[0]

            #Get number of days between start date and end date
            ndays = (atm_c.index[-1] - atm_c.index[0]).days + 1
            #Calculate end of term position status
            chain['ATM_PL'] = ((chain['CLOSE_CA'] - aco) + (chain['CLOSE_PA'] - apo)) * r1 * lot_size        #BUY ATM STRIKE 2 LOTS
            chain['SHT_PL'] = ((sco - chain['CLOSE_CS']) + (spo - chain['CLOSE_PS'])) * r2 * lot_size        #SHORT ONE STRIKE AWAY FROM ATM 3 LOTS
            chain['LNG_PL'] = ((chain['CLOSE_CL'] - lco) + (chain['CLOSE_PL'] - lpo)) * r3 * lot_size        #BUY ONE STRIKE AWAY FROM SHORT STRIKE 1 LOT
            chain['DPL'] =  chain['ATM_PL'] +  chain['SHT_PL'] + chain['LNG_PL']
            chain['PL_OPEN'] = 0
            chain['PL_LOW'] = chain['DPL']
            chain['PL_HIGH'] = chain['DPL']
            chain['PL_CLOSE'] = chain['DPL']
            rchain = chain.resample(f'{ndays}D').agg(Defaults.DOUBLE_RATIO_CONVERSION)
            rchain['PL_LO_IDX'] = chain.reset_index()['DPL'].idxmin()
            rchain['PL_HI_IDX'] = chain.reset_index()['DPL'].idxmax()
            rchain['MAX_LOSS'] = ((aco + apo) * r1 * lot_size) + ((lco + lpo) * r3 * lot_size) - ((sco + spo) * r2 * lot_size)
            sd = sdi.set_index('TIMESTAMP').resample(f'{ndays}D').agg(Defaults.CONVERSION)
            sd['PCR'] = self.oi_pcr[(self.oi_pcr['TIMESTAMP'] == st) & (self.oi_pcr['EXPIRY_DT'] == nd)]['PCR'].iloc[0]
            return sd.merge(rchain, how='inner', left_index=True, right_index=True)
        except Exception as ex:
            print_exception(ex)
            return None

    def double_ratio_spread_days_before_expiry(self, symbol, instrument, lot_size, margin_per_lot, brokerage, r1=2, r2=3, days_before=2, n_expiry=10):
        """ ratio spread shortly before expiry
            strategy is to buy two losts of CE & PE ATM
            sell 3 lots of next immediate OTM's respectively
            buy 1 lot of next immedate OTM's (after the sold OTM) respectively
        Parameters:
            symbol - 'BANKNIFTY', 'NIFTY', 'HDFC' etc
            instrument - OPTIDX or STKIDX
            lot_size - size of lot
            margin_per_lot - margin per lot
            brokerage - total brokerage for the strategy
            r1 - first part of ratio write - long leg - shall be starting from 2
            r2 - second part of ratio write - short leg - shall be r1 + at least 1
            days_before - number of days before expiry the trade initiated
            n_expiry = number of expirys to be considered
            """
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            self.expirys['START_DT'] = self.expirys['EXPIRY_DT'] - pd.Timedelta(f'{days_before}Day')
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.double_ratio_spread_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], lot_size=lot_size, r1=r1, r2=r2))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_double_ratio(sdf, lot_size, r2, margin_per_lot, brokerage)
            self.write_to_excel(f'{symbol}_RATIO_SPREAD_{days_before}_BE_{r1}-{r2}-{r2-r1}_{n_expiry}')
        except Exception as ex:
            print_exception(ex)
            return None

    def e2e_double_ratio_spread(self, symbol, instrument, lot_size, margin_per_lot, brokerage, r1=2, r2=3, n_expiry=10):
        """ ratio spread shortly before expiry
            strategy is to buy two losts of CE & PE ATM
            sell 3 lots of next immediate OTM's respectively
            buy 1 lot of next immedate OTM's (after the sold OTM) respectively
        Parameters:
            symbol - 'BANKNIFTY', 'NIFTY', 'HDFC' etc
            instrument - OPTIDX or STKIDX
            lot_size - size of lot
            margin_per_lot - margin per lot
            brokerage - total brokerage for the strategy
            r1 - first part of ratio write - long leg - shall be starting from 2
            r2 - second part of ratio write - short leg - shall be r1 + at least 1
            n_expiry = number of expirys to be considered
            """
        try:
            self.prepare_strategy(symbol, instrument, n_expiry)
            expg = self.expirys.groupby(['START_DT', 'EXPIRY_DT'])
            sdf = expg.apply(lambda x: self.double_ratio_spread_builder(x['START_DT'].iloc[0], x['EXPIRY_DT'].iloc[0], lot_size=lot_size, r1=r1, r2=r2))
            sdf = sdf.reset_index().drop(['TIMESTAMP'], axis=1)
            self.strategy_df = self.calculate_profit_double_ratio(sdf, lot_size, r2, margin_per_lot, brokerage)
            self.write_to_excel(f'{symbol}_E2E_RATIO_SPREAD_{r1}-{r2}-{r2-r1}_{n_expiry}')
        except Exception as ex:
            print_exception(ex)
            return None

    def e2e_ratio_write_at_max_oi(self, symbol, instrument, lot_size, margin_per_lot, brokerage, n_expiry=10):
        """
        Ratio write: buy a strike away from atm in any direction and sell more number of strikes next to it having lower premiums.
        If strike 120CE is bought for 24 premium then sell 3X strike 140CE at 12 premium
        """

if __name__ == '__main__':
    op = OptionsBT()