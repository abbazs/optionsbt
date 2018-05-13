from datetime import datetime, date, timedelta
import logging
import sys
import os
import inspect

logger = None


def start_logger():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter(
        '%(asctime)s|[%(levelname)8s]|[%(module)s.%(name)s.%(funcName)s]|%(lineno)4s|%(message)s')

    file_handler = logging.FileHandler(filename='OptionsBT_{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')),
                                       mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)


def print_exception(e: object):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    short_message = 'Exception at {} - {} - {}\nCalling Function: {}'.format(exc_type,
                                                                             file_name,
                                                                             exc_tb.tb_lineno,
                                                                             inspect.stack()[1][3])
    message = '{}\n{}'.format(short_message, str(e))
    logger.debug(message)


def get_current_date():
    if datetime.today().hour < 17:
        dt = date.today() - timedelta(days=1)
        return datetime.combine(dt, datetime.min.time())
    else:
        return datetime.combine(date.today(), datetime.min.time())


def fix_start_and_end_date(start_date, end_date):

    if end_date is None:
        end_date = start_date
    else:
        if start_date > end_date:
            start_date, end_date = end_date, start_date

    return start_date, end_date

# Initialize Logger
start_logger()
