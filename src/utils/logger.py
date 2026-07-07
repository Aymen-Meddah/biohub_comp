import logging

from pathlib import Path


class Logger:

    def __init__(

        self,

        log_file

    ):

        Path(log_file).parent.mkdir(

            parents=True,

            exist_ok=True

        )

        self.logger = logging.getLogger(

            "CellTracking"

        )

        self.logger.setLevel(

            logging.INFO

        )

        self.logger.handlers.clear()

        formatter = logging.Formatter(

            "%(asctime)s | %(levelname)s | %(message)s"

        )

        file_handler = logging.FileHandler(

            log_file

        )

        file_handler.setFormatter(

            formatter

        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(

            formatter

        )

        self.logger.addHandler(

            file_handler

        )

        self.logger.addHandler(

            console_handler

        )

    def info(

        self,

        message

    ):

        self.logger.info(

            message

        )

    def warning(

        self,

        message

    ):

        self.logger.warning(

            message

        )

    def error(

        self,

        message

    ):

        self.logger.error(

            message

        )

    def debug(

        self,

        message

    ):

        self.logger.debug(

            message
        )