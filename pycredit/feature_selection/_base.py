from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np
from pycredit.utils.utils_func import _check_attrs, to_polars_dataframe
import polars as pl

class FeatureSelector(metaclass=ABCMeta):
    """
    特征选择器的基类，定义了特征选择器的基本接口

    Parameters
    ----------
    verbose : bool, default=False
        是否打印详细信息。

    Attributes
    ----------
    input_features\_ : np.ndarray
        输入特征列表。
    feature_mask\_ : List
        特征掩码，指示哪些特征被选择。
    score\_ : np.ndarray
        特征选择器的分数列表，用于筛选特征。

    """

    def __init__(self, verbose: bool=False) -> None:
        self.input_features_: np.ndarray = None
        self.feature_mask_: List = None
        self.score_: np.ndarray = None
        self._verbose = verbose

    @abstractmethod
    def fit(self, X, y=None):
        """
        拟合特征选择器

        Parameters
        ----------
        X : array-like or DataFrame
            待筛选数据。
        y : array-like, optional
            目标变量，监督式特征选择时使用。

        Returns
        -------
        self: object
            返回拟合后的特征选择器实例。
        """
        pass

    @_check_attrs('input_features_', 'feature_mask_')
    def transform(self, X) -> pl.DataFrame:
        """
        根据拟合结果转换特征

        Parameters
        ----------
        X : array-like or DataFrame
            待筛选数据。

        Returns
        -------
        X_: pl.DataFrame
            筛选后的数据。
        """
        X = to_polars_dataframe(X)
        output_features_ = self.input_features_[self.feature_mask_]
        X_ = X.select(output_features_)
        return X_

    def fit_transform(self, X, y=None) -> pl.DataFrame:
        """
        拟合并转换特征

        Parameters
        ----------
        X : array-like or DataFrame
            待筛选数据。
        y : array-like, optional
            标签（可选）。

        Returns
        -------
        X_: pl.DataFrame
            筛选后的数据。
        """
        if y is not None:
            self.fit(X, y)
        else:
            self.fit(X)

        X_ = self.transform(X)
        return X_

    @_check_attrs('input_features_', 'feature_mask_')
    def get_feature_names_out(self) -> np.ndarray:
        '''
        获取转换后的特征名称

        Returns
        -------
        output_features_: np.ndarray
            筛选后特征名称列表。
        '''
        output_features_ = self.input_features_[self.feature_mask_]
        return output_features_

    @_check_attrs('input_features_')
    def get_feature_names_in(self) -> np.ndarray:
        '''
        获取输入特征名称

        Returns
        -------
        input_features_: np.ndarray
            输入特征名称列表
        '''
        return self.input_features_

    @_check_attrs('input_features_')
    def get_feature_num_in(self) -> int:
        """
        获取输入特征数量

        Returns
        -------
        input_feature_num: int
            输入特征的数量
        """
        return len(self.input_features_)

    @_check_attrs('feature_mask_')
    def get_feature_num_out(self) -> int:
        """
        获取输出特征数量

        Returns
        -------
        output_feature_num: int
            筛选后特征的数量
        """
        return sum(self.feature_mask_)

    @_check_attrs('score_')
    def get_score(self) -> np.ndarray:
        '''
        获取特征选择器的分数列表

        Returns
        -------
        score_: np.ndarray
            特征选择器的分数列表
        '''
        return self.score_

