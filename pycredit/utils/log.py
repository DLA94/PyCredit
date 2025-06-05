import sys
import logging
from functools import wraps

# 日志类
class Logger(object):
    """
    日志类

    Parameters
    ----------
    name: str, optional
        日志器名称，默认为 'log'。
    log_file: str, optional
        日志文件路径，若为 None 则只输出到终端。
    level: int, default=logging.INFO
        日志级别，默认为 logging.INFO。
    """

    def __init__(self, name='log', log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # 如果指定了日志文件，则添加文件处理器
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # 装饰器方法(方法调试)
    def __call__(self, func):
        """
        使 Logger 实例可作为装饰器，自动记录函数调用、参数、返回值和异常。

        Parameters
        ----------
        func:
            被装饰的函数。

        Returns
        -------
        result:
            函数返回值
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.logger.info(f'调用函数: {func.__name__} 参数: {args} {kwargs}')
            try:
                result = func(*args, **kwargs)
                self.logger.info(f'函数 {func.__name__} 返回: {result}')
                return result
            except Exception as e:
                self.logger.error(f'函数 {func.__name__} 异常: {e}', exc_info=True)
                raise
        return wrapper

    def log(self, level, message)-> None:
        """
        按不同日志级别输出带颜色的日志信息。

        Parameters
        ----------
        level: str
            日志级别，如 'info', 'debug', 'warning', 'error', 'critical'。
        message: str
            日志内容。
        """
        color_map = {
            'info': '\033[0m',       # 默认颜色
            'debug': '\033[90m',     # 浅灰色
            'warning': '\033[33m',   # 黄色
            'error': '\033[31m',     # 红色
            'critical': '\033[41m',  # 红底
        }
        reset = '\033[0m'
        color = color_map.get(level, '')
        colored_message = f"{color}{message}{reset}"

        if level == 'info':
            self.logger.info(colored_message)
        elif level == 'debug':
            self.logger.debug(colored_message)
        elif level == 'warning':
            self.logger.warning(colored_message)
        elif level == 'error':
            self.logger.error(colored_message)
        elif level == 'critical':
            self.logger.critical(colored_message)
        else:
            raise ValueError(f'不支持的日志级别: {level}')

