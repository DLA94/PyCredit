from pycredit.feature_selection._base import FeatureSelector
from pycredit.utils.utils_func import to_polars_dataframe
import numpy as np
import polars as pl
from pycredit.utils.log import Logger

class NonNullFeatureSelector(FeatureSelector):
    '''
    缺失值特征选择器，用于识别数据集中缺失值比例超过阈值的特征列

    Parameters
    ----------
    verbose : bool, default=False
        是否打印日志信息，默认为False
    null_threshold : float, default=0.8
        缺失值比例阈值，超过该比例的特征将被剔除，默认为0.8

    Attributes
    ----------
    input_features\_ : np.ndarray
        输入特征列表。
    feature_mask\_ : List
        特征掩码，指示哪些特征被选择。
    score\_ : np.ndarray
        特征选择器的分数列表，用于筛选特征。
    '''

    def __init__(self,
                 verbose: bool = False,
                 null_threshold: float=0.8) -> None:
        super().__init__(verbose=verbose)

        self._null_threshold = null_threshold

    def fit(self, X) -> object:
        """
        拟合特征选择器，识别缺失值比例超过阈值的特征列

        Parameters
        ----------
        X: array-like or DataFrame
            待筛选数据

        Returns
        -------
        self: NonNullFeatureSelector
            返回拟合后的特征选择器实例。
        """
        X = to_polars_dataframe(X)

        # 获取输入特征列表
        self.input_features_ = np.array(X.columns)

        # 计算每个特征的缺失率
        self.score_ = X.select(
            [(pl.col(col).is_null().sum()/pl.col(col).count()).alias(col) for col in self.input_features_]
        ).to_numpy()[0]

        self.feature_mask_ = (self.score_ <= self._null_threshold).tolist()

        if self._verbose:
            Logger().log('info', f'原始特征列数量：{self.get_feature_num_in()}')
            Logger().log('info', f'缺失率低于{self._null_threshold * 100}%特征列数量：{self.get_feature_num_out()}')

        return self