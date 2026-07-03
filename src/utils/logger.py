import logging
from os import path
from pathlib import Path 

class Logger:
    def __init__(self ,log_dir="logs", log_name="train.log" , level=logging.INFO):
        self.log_dir = path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_name
        self.logger = logging.getLogger("BioHub")
        self.logger.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.log_file ,mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)
    def error(self, message):
        self.logger.error(message)
    def debug(self, message):
        self.logger.debug(message)
    def critical(self, message):
        self.logger.critical(message)
