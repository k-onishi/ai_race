# -*- coding: utf-8 -*-
import logging


def get_logger(name=None, level=logging.INFO, log_filename=None):
    if name is None:
        logger = logger.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(
        logging.StreamHandler()
    )
    if log_filename is not None:
        logger.addHandler(
            logging.FileHandler(log_filename)
        )
    return logger

logger = get_logger(
        name="logger",
        level=logging.DEBUG,
        log_filename="/tmp/logger.log"
    )
