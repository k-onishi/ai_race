# -*- coding: utf-8 -*-
import logging


def get_logger(name=None, level=logging.INFO, log_filename=None):
    if name is None:
        logger = logger.getLogger(__name__)
    else:
        logger = logging.getLogger(name)
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    if log_filename is not None:
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger

logger = get_logger(
        name="logger",
        level=logging.DEBUG,
        log_filename="/tmp/logger.log"
    )
