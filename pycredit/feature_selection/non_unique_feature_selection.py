from pycredit.feature_selection._base import FeatureSelector
from pycredit.utils.utils_func import to_polars_dataframe
import polars as pl
import numpy as np
from pycredit.utils.log import Logger

class NonUniqueFeatureSelector(FeatureSelector):
    '''
    非唯一特征选择器，用于识别数据集中非唯一值的特征列

    Parameters
    ----------
    verbose : bool, default=False
        是否打印日志信息，默认为 False

    Attributes
    ----------
    input_features\_ : np.ndarray
        输入特征列表。
    feature_mask\_ : List
        特征掩码，指示哪些特征被选择。
    score\_ : np.ndarray
        特征选择器的分数列表，用于筛选特征。
    '''
    def __init__(self, verbose=False) -> None:
        super().__init__(verbose=verbose)

    def fit(self, X) -> object:
        """
        拟合特征选择器，识别非唯一值的特征列

        Parameters
        ----------
        X: array-like or DataFrame
            待筛选数据

        Returns
        -------
        self: NonUniqueFeatureSelector
            返回拟合后的特征选择器实例。
        """
        X = to_polars_dataframe(X)

        # 获取输入特征列表
        self.input_features_ = np.array(X.columns)

        # 计算特征评分
        self.score_ = X.select(
            [pl.col(col).n_unique() for col in self.input_features_]
        ).to_numpy()[0]

        # 生成特征掩码，标记非唯一特征
        self.feature_mask_ = (self.score_ > 1).tolist()

        if self._verbose:
            Logger().log('info', f'原始特征列数量：{self.get_feature_num_in()}')
            Logger().log('info', f'非唯一值特征列数量：{self.get_feature_num_out()}')

        return self