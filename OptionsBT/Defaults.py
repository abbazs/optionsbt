WEEKLY_STRADDLE_COLUMNS = ['TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'STRIKE', 'OPEN_C', 'HIGH_C', 'LOW_C',
                           'CLOSE_C', 'H-O_C',
                           'L-O_C', 'C-O_C', 'OPEN_P', 'HIGH_P', 'LOW_P', 'CLOSE_P', 'H-O_P', 'L-O_P', 'C-O_P']

OPTIONS_COLUMNS = ['TIMESTAMP', 'EXPIRY_DT', 'SYMBOL', 'STRIKE_PR', 'OPTION_TYP', 'OPEN', 'HIGH', 'LOW', 'CLOSE']

CONVERSION = {'OPEN': 'first', 'HIGH': 'max', 'LOW': 'min', 'CLOSE': 'last'}