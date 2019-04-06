WEEKLY_STRADDLE_COLUMNS = ['TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'STRIKE', 'OPEN_C', 'HIGH_C', 'LOW_C',
                           'CLOSE_C', 'H-O_C',
                           'L-O_C', 'C-O_C', 'OPEN_P', 'HIGH_P', 'LOW_P', 'CLOSE_P', 'H-O_P', 'L-O_P', 'C-O_P']

OPTIONS_COLUMNS = ['TIMESTAMP', 'EXPIRY_DT', 'SYMBOL', 'STRIKE_PR', 'OPTION_TYP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'OPEN_INT', 'CHG_IN_OI']

CANDLE_COLUMNS = ['OPEN', 'HIGH', 'LOW', 'CLOSE']

CALL_CANDLE_COLUMNS = [f'{x}_C' for x in CANDLE_COLUMNS]

PUT_CANDLE_COLUMNS = [f'{x}_P' for x in CANDLE_COLUMNS]

PL_CANDLE_COLUMNS = [f'PL_{x}' for x in CANDLE_COLUMNS]

CONSTANT_COLUMNS = ['LOT_SIZE', 'NUM_LOTS', 'TOTAL_LOTS', 'MARGIN_PER_LOT', 'MARGIN_REQUIRED', 'STOP_LOSS', 'STOP_LOSS_TRIGGER']

PL_NET_COLUMNS = ['NET_PL_LO', 'NET_PL_HI', 'NET_PL_CL']

STOP_LOSS_TRUTH_COLUMNS = ['STOP_LOSS_HIT', 'SL_HI_GT_CL']

NET_COLUMNS = ['NET', '%', 'LOSE_COUNT', 'CUM_LOSE_COUNT', 'CUM_NET']

CONVERSION = {'SYMBOL':'first', 'OPEN':'first', 'HIGH':'max', 'LOW':'min', 'CLOSE':'last'}

CHAIN_CONVERSION = {'OPEN_C':'first', 'HIGH_C':'max', 'LOW_C':'min', 'CLOSE_C':'last', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last'}

STRADDLE_CONVERSION = {'OPEN_C':'first', 'HIGH_C': 'max', 'LOW_C':'min', 'CLOSE_C':'last', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last','PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}

STRANGLE_CONVERSION = {'STRIKE_C':'first', 'OPEN_C':'first', 'HIGH_C':'max', 'LOW_C':'min', 'CLOSE_C':'last', 'STRIKE_P':'first', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last', 'PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}

IRON_BUTTERFLY_CONVERSION = {'OPEN_C':'first', 'HIGH_C':'max', 'LOW_C':'min', 'CLOSE_C':'last', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last', 'STRIKE_CL':'first', 'OPEN_CL':'first', 'HIGH_CL':'max', 'LOW_CL':'min', 'CLOSE_CL':'last', 'STRIKE_PR':'first', 'OPEN_PR':'first', 'HIGH_PR':'max', 'LOW_PR':'min', 'CLOSE_PR':'last', 'PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}

CALENDAR_SPREAD_CONVERSION = {'OPEN_C':'first', 'HIGH_C':'max', 'LOW_C':'min', 'CLOSE_C':'last', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last', 'OPEN_CN':'first', 'HIGH_CN':'max', 'LOW_CN':'min', 'CLOSE_CN':'last', 'OPEN_PN':'first', 'HIGH_PN':'max', 'LOW_PN':'min', 'CLOSE_PN':'last', 'PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}

DOUBLE_RATIO_COLUMNS = ['STRIKE_ATM', 'OPEN_CA','HIGH_CA', 'LOW_CA', 'CLOSE_CA', 'OPEN_PA', 'HIGH_PA', 'LOW_PA', 'CLOSE_PA', 'STRIKE_CS', 'OPEN_CS', 'HIGH_CS','LOW_CS', 'CLOSE_CS', 'STRIKE_PS', 'OPEN_PS', 'HIGH_PS', 'LOW_PS', 'CLOSE_PS', 'STRIKE_CL', 'OPEN_CL', 'HIGH_CL','LOW_CL', 'CLOSE_CL', 'STRIKE_PL', 'OPEN_PL', 'HIGH_PL', 'LOW_PL', 'CLOSE_PL']

DOUBLE_RATIO_CONVERSION =  {'STRIKE_ATM':'first', 'OPEN_CA':'first', 'HIGH_CA':'max', 'LOW_CA':'min', 'CLOSE_CA':'last', 'OPEN_PA':'first', 'HIGH_PA':'max', 'LOW_PA':'min', 'CLOSE_PA':'last', 'STRIKE_CS':'first', 'OPEN_CS':'first', 'HIGH_CS':'max','LOW_CS':'min', 'CLOSE_CS':'last', 'STRIKE_PS':'first', 'OPEN_PS':'first', 'HIGH_PS':'max', 'LOW_PS':'min', 'CLOSE_PS':'last', 'STRIKE_CL':'first', 'OPEN_CL':'first', 'HIGH_CL':'max','LOW_CL':'min', 'CLOSE_CL':'last', 'STRIKE_PL':'first', 'OPEN_PL':'first', 'HIGH_PL':'max', 'LOW_PL':'min', 'CLOSE_PL':'last', 'PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}

RATIO_WRITE_AT_MAX_OI_COLUMNS = ['STRIKE_C', 'OPEN_C', 'HIGH_C', 'LOW_C', 'CLOSE_C', 'STRIKE_P', 'OPEN_P', 'HIGH_P', 'LOW_P', 'CLOSE_P', 'STRIKE_CS', 'OPEN_CS', 'HIGH_CS','LOW_CS', 'CLOSE_CS', 'STRIKE_PS', 'OPEN_PS', 'HIGH_PS', 'LOW_PS', 'CLOSE_PS']

RATIO_WRITE_AT_MAX_OI_CONVERSION =  {'STRIKE_C':'first', 'OPEN_C':'first', 'HIGH_C':'max', 'LOW_C':'min', 'CLOSE_C':'last', 'STRIKE_P':'first', 'OPEN_P':'first', 'HIGH_P':'max', 'LOW_P':'min', 'CLOSE_P':'last', 'STRIKE_CS':'first', 'OPEN_CS':'first', 'HIGH_CS':'max','LOW_CS':'min', 'CLOSE_CS':'last', 'STRIKE_PS':'first', 'OPEN_PS':'first', 'HIGH_PS':'max', 'LOW_PS':'min', 'CLOSE_PS':'last', 'PL_OPEN':'first', 'PL_HIGH':'max', 'PL_LOW':'min', 'PL_CLOSE':'last'}