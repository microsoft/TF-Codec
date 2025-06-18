# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pathlib
import logging


_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_logger(logger, level=None, log_format=_LOG_FORMAT):

    handler_stderr = logging.StreamHandler()
    handler_stderr.setFormatter(logging.Formatter(log_format))
    handler_stderr.setLevel(logging.NOTSET)

    logger.handlers = [handler_stderr]  # Reset the handlers!

    if level is not None:
        logger.setLevel(level)

    return logger


def _log_to_file(logger, fname, level=logging.NOTSET, log_format=_LOG_FORMAT):
    "Add handler that writes log to the file. Creates log path if it does not exist."

    log_dir = os.path.dirname(fname)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    handler_file = logging.FileHandler(fname)
    handler_file.setFormatter(logging.Formatter(log_format))
    handler_file.setLevel(level)

    logger.addHandler(handler_file)


def get_logger(name, level=logging.DEBUG):
    "Get stderr logger for AzureML tasks"
    logger = logging.getLogger(name)
    type(logger).log_to_file = _log_to_file
    logger.setLevel(level)
    return logger


_configure_logger(logging.root)