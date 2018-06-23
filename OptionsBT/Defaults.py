WEEKLY_STRADDLE_COLUMNS = ['TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'STRIKE', 'OPEN_C', 'HIGH_C', 'LOW_C',
                           'CLOSE_C', 'H-O_C',
                           'L-O_C', 'C-O_C', 'OPEN_P', 'HIGH_P', 'LOW_P', 'CLOSE_P', 'H-O_P', 'L-O_P', 'C-O_P']

OPTIONS_COLUMNS = ['TIMESTAMP', 'EXPIRY_DT', 'SYMBOL', 'STRIKE_PR', 'OPTION_TYP', 'OPEN', 'HIGH', 'LOW', 'CLOSE']

CANDLE_COLUMNS = ['OPEN', 'HIGH', 'LOW', 'CLOSE']

CALL_CANDLE_COLUMNS = [f'{x}_C' for x in CANDLE_COLUMNS]

PUT_CANDLE_COLUMNS = [f'{x}_P' for x in CANDLE_COLUMNS]

PL_CANDLE_COLUMNS = [f'PL_{x}' for x in CANDLE_COLUMNS]

CONSTANT_COLUMNS = ['LOT_SIZE', 'NUM_LOTS', 'TOTAL_LOTS', 'MARGIN_PER_LOT', 'MARGIN_REQUIRED', 'STOP_LOSS', 'STOP_LOSS_TRIGGER']

PL_NET_COLUMNS = ['NET_PL_LO', 'NET_PL_HI', 'NET_PL_CL']

STOP_LOSS_TRUTH_COLUMNS = ['STOP_LOSS_HIT', 'SL_HI_GT_CL']

NET_COLUMNS = ['NET', '%', 'LOSE_COUNT', 'CUM_LOSE_COUNT', 'CUM_NET']

CONVERSION = {'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last'}

CHAIN_CONVERSION = {'OPEN_C': 'first', 'HIGH_C': 'max', 'LOW_C': 'min', 'CLOSE_C': 'last', 'OPEN_P': 'first', 'HIGH_P': 'max', 'LOW_P': 'min', 'CLOSE_P': 'last'}

STRADDLE_CONVERSION = {'OPEN_C': 'first', 'HIGH_C': 'max', 'LOW_C': 'min', 'CLOSE_C': 'last', 'OPEN_P': 'first', 'HIGH_P': 'max', 'LOW_P': 'min', 'CLOSE_P': 'last','PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}