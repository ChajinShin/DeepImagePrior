import time
import sys
import logging
import pandas as pd
from collections import OrderedDict
from pathlib2 import Path
from typing import Union


class ProcessBar(object):
    def __init__(self, max_iter, prefix='', suffix='', bar_length=50):
        self.max_iter = max_iter
        self.prefix = prefix
        self.suffix = suffix
        self.bar_length = bar_length
        self.iteration = 0

    def step(self, other_info: str=None):
        self.iteration += 1

        percent = 100 * self.iteration / self.max_iter
        filled_length = int(round(self.bar_length * self.iteration) / self.max_iter)
        bar = '#' * filled_length + '-' * (self.bar_length - filled_length)
        msg = '\r{} [{}] {:.1f}% {}'.format(self.prefix, bar, percent, self.suffix)
        if other_info is not None:
            msg = msg + "  |   " + other_info
        sys.stdout.write(msg)
        if self.iteration == self.max_iter:
            sys.stdout.write('\n')
        sys.stdout.flush()


class ElapsedTimeProcess(object):
    def __init__(self, max_iter: int, start_iter: int = 0, output_type: str = 'summary_with_str'):
        self.max_iter = max_iter
        self.current_iter = start_iter
        if output_type not in ['seconds', 'summary', 'summary_with_str']:
            raise ValueError("Unknown type '{}'.".format(self.output_type))
        self.output_type = output_type
        self.t1 = 0
        self.t2 = 0

    def start(self):
        self.t1 = time.time()

    def end(self):
        self.t2 = time.time()

        # 남은 시간 (Second)
        eta = (self.t2 - self.t1) * (self.max_iter - self.current_iter)
        self.current_iter += 1

        if self.output_type == 'seconds':
            return eta
        elif self.output_type == 'summary':
            return self._summary(eta, with_str=False)
        elif self.output_type == 'summary_with_str':
            return self._summary(eta, with_str=True)
        else:
            raise ValueError("Unknown type '{}'.".format(self.output_type))

    def _summary(self, eta, with_str=True):
        elapsed_time_dict = self._calculate_summary(eta)
        if with_str:
            return self._to_string(elapsed_time_dict)
        else:
            return elapsed_time_dict

    @staticmethod
    def _calculate_summary(eta):
        elapsed_time_dict = OrderedDict()

        # days
        eta_days = int(eta // (24 * 3600))
        if eta_days != 0:
            elapsed_time_dict['eta_days'] = eta_days

        # hours
        eta_hours = int((eta // 3600) % 24)
        if eta_hours != 0:
            elapsed_time_dict['eta_hours'] = eta_hours

        # minutes
        eta_minutes = int((eta // 60) % 60)
        if eta_minutes != 0:
            elapsed_time_dict['eta_minutes'] = eta_minutes

        # seconds
        elapsed_time_dict['eta_seconds'] = int(eta % 60)

        return elapsed_time_dict

    @staticmethod
    def _to_string(elapsed_time_dict):
        output = ''
        for key, value in elapsed_time_dict.items():
            if key == 'eta_days':
                output += '{} days '.format(value)
            elif key == 'eta_hours':
                output += '{} h '.format(value)
            elif key == 'eta_minutes':
                output += '{} m '.format(value)
            elif key == 'eta_seconds':
                output += '{} s'.format(value)
            else:
                raise KeyError('Some key has mismatched name')
        return output


class Logger(object):
    def __init__(self, logging_file_dir: Path, log_level: str = 'INFO'):
        self.logger = logging.getLogger('log')
        self.logger.setLevel(self._get_log_level(log_level))
        handler = logging.FileHandler(logging_file_dir)
        self.logger.addHandler(handler)

    def change_log_level(self, log_level: str):
        self.logger.setLevel(log_level)

    def debug(self, log: str):
        self.logger.debug(log)

    def info(self, log: str):
        self.logger.info(log)

    def warning(self, log: str):
        self.logger.warning(log)

    def error(self, log):
        self.logger.error(log)

    def critical(self, log):
        self.logger.critical(log)

    @staticmethod
    def _get_log_level(log_level: str):
        level_dict = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return level_dict[log_level]


class CSV_Recorder(object):
    def __init__(self):
        self.csv_recorder = pd.DataFrame()

    def load_csv(self, path: Path, index_col: Union[None, int] = None):
        self.csv_recorder = pd.read_csv(path, index_col=index_col)

    def write_data(self, row: Union[int, str], col: Union[int, str], data):
        if isinstance(row, str) and isinstance(col, str):
            self.csv_recorder.loc[row, col] = data
        elif isinstance(row, int) and isinstance(col, int):
            self.csv_recorder.iloc[row, col] = data

    def reset_with_offset(self, offset: int):
        """
        This function remain index 0 to offset - 1.
        All other rows will be deleted.
        """
        length = self.csv_recorder.shape[0]
        index = self.csv_recorder.index
        self.csv_recorder = self.csv_recorder.drop(list(index[offset: length]), axis=0)

    def to_csv(self, path: Union[str, Path]):
        self.csv_recorder.to_csv(path)

