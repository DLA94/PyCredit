'''
通用工具函数
'''
from functools import wraps
from pycredit.utils.log import Logger
import sys
import polars as pl
import pandas as pd
import numpy as np
import math

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


def score_transfer(prob, pdo=20, base_odds=20, base_score=600):
    """
    转换评分

    Parameters
    ----------
    prob: array-like (n_samples,)
        预测概率
    pdo: float
        两倍odds增长评分
    base_odds: float
        基准odds
    base_score: float
        基准评分

    Returns
    -------
    score: array-like (n_samples,)
        转换后的评分
    """
    B = pdo / math.log(2)
    A = base_score + B * math.log(base_odds)
    score = A - B * (np.log(prob) / (1 - np.log(prob)))
    return score