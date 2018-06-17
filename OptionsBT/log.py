from datetime import datetime
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

# Initialize Logger
start_logger()
