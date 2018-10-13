# -*- coding: utf-8 -*-

import logging

from .h5rw import *
from .plot import *
from data_preprocessing import *

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

logger = None


def init_logging(params):
    global logger
    logging.basicConfig(level=INFO)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                  "%H:%M:%S")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(params.logging.level)
    console_handler.setFormatter(formatter)

    # create a file handler
    handler = logging.FileHandler(params.logging.log_file_path)
    handler.setLevel(params.logging.level)
    handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(params.logging.level)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)
