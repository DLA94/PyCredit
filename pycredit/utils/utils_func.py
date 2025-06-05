'''
通用工具函数
'''
from functools import wraps
from pycredit.utils.log import Logger
import sys
import polars as pl
import pandas as pd
import numpy as np

def _check_attrs(*attrs):
    """
    检查 self 是否包含指定属性且不为 None 的装饰器

    Parameters
    ----------
    *attrs:
        属性列表
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            for attr in attrs:
                if not hasattr(self, attr) or getattr(self, attr) is None:
                    Logger().log('error', f"属性 '{attr}' 未设置或为 None，请先调用相应方法进行初始化。")
                    sys.exit(-1)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def to_polars_dataframe(data: [pd.DataFrame, np.ndarray, pl.DataFrame]) -> pl.DataFrame:
    """
    将输入数据转换为 Polars DataFrame

    Parameters
    ----------
    data: array-like or DataFrame
        输入数据

    Returns
    -------
    data: pl.DataFrame
        转换后的 Polars DataFrame
    """
    if isinstance(data, pl.DataFrame):
        return data
    elif isinstance(data, (list, tuple, np.ndarray)):
        return pl.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        return pl.DataFrame(data)
    else:
        Logger().log("error", f"输入数据为{type(data)}，无法转换为 Polars DataFrame。请提供 Pandas DataFrame、Numpy 数组或 Polars DataFrame。")
        sys.exit(-1)