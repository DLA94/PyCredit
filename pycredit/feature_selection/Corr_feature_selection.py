import numpy as np
from pycredit.feature_selection._base import FeatureSelector
from typing import Optional, List
from pycredit.utils.utils_func import to_polars_dataframe
import polars as pl
from pycredit.utils.log import Logger

class CorrFeatureSelector(FeatureSelector):
    '''
    相关性特征选择。基于特征之间的Person相关性进行筛选。
    使用启发式方式进行特征选择，优先剔除相关性高于阈值且 `column_priority` 中优先级较低的特征。

    Parameters
    ----------
    verbose : bool, default=False
        是否打印详细信息。
    threshold : float, default=0.8
        相关性阈值，低于此值的特征将被筛选掉。
    heuristic_mode : str, default=global
        特征选择模式。`heuristic_mode='local'` 时，结果可能不是全局最优。
    fill_null : str, default='mean'
        空值填充策略，可选 'mean', 'median', 'min', 'max'。

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
                 verbose: bool=False,
                 threshold: float=0.8,
                 heuristic_mode: str='local',
                 fill_null: str='mean') -> None:

        super().__init__(verbose=verbose)

        self._threshold = threshold
        self._heuristic_mode = heuristic_mode
        self._fill_null = fill_null

    def _corr_selection(self) -> None:
        '''
        根据相关性矩阵和列优先级进行特征选择。

        Returns
        -------
        None
        '''
        # 筛选高于阈值特征列关系对
        candidate_feature_pairs = []
        for i in range(self.get_feature_num_in()):
            for j in range(i + 1, self.get_feature_num_in()):
                if abs(self.score_[i][j]) > self._threshold:
                    candidate_feature_pairs.append([i, j, abs(self.score_[i][j])])

        candidate_feature_pairs = sorted(candidate_feature_pairs, key=lambda x: x[2], reverse=True)

        for (i, j, _) in candidate_feature_pairs:
            if self.feature_mask_[i]:
                fe_i = self._column_priority.index(self.input_features_[i])
            else:
                continue

            if self.feature_mask_[j]:
                fe_j = self._column_priority.index(self.input_features_[j])
            else:
                continue

            if fe_i > fe_j:
                self.feature_mask_[j] = False
            else:
                self.feature_mask_[i] = False

    def fit(self, X, column_priority: Optional[List[str]]=None) -> object:
        '''
        训练相关性特征选择器。

        Parameters
        ----------
        X : array-like or DataFrame
            待筛选数据。

        column_priority : Optional[List[str]], default=None
            特征列的优先级列表。如未提供，则默认按列名顺序。

        Returns
        -------
        self: CorrFeatureSelector
            返回CorrFeatureSelector实例。

        note
        ----
            输入数据特征列需为数值型！

        '''
        # 校验输入数据类型
        X = to_polars_dataframe(X)

        # 校验输入数据类型
        if not all(dtype in pl.NUMERIC_DTYPES for dtype in X.schema.values()):
            raise TypeError("输入数据特征列需为数值型！")

        if column_priority is None:
            self._column_priority = X.columns
        else:
            self._column_priority = list(column_priority)

        # 校验column_priority与X.columns是否一致
        if set(self._column_priority) != set(X.columns):
            raise ValueError("`column_priority` 中的列名必须与输入数据的列名一致！")

        self.input_features_ = np.array(X.columns)
        self.feature_mask_ = np.ones_like(self.input_features_, dtype=bool).tolist() # 特征掩码初始化

        # 缺失值填充
        X = X.fill_null(strategy=self._fill_null)

        # 计算相关性矩阵
        if self._heuristic_mode == 'global':
            self.score_ = np.corrcoef(X.to_numpy(), rowvar=False)
            self._corr_selection()

        elif self._heuristic_mode == 'local':
            self.score_ = np.empty(shape=(self.get_feature_num_in(), self.get_feature_num_in()))

            for i, fea_i in enumerate(self._column_priority):
                temp_cols = self._column_priority[i+1:]
                for j, fea_j in enumerate(temp_cols):
                    corr = X.select(pl.corr(pl.col(fea_i), pl.col(fea_j))).to_numpy()[0]
                    self.score_[i][j]= self.score_[j][i] = corr
                    if abs(corr) > self._threshold:
                        self._column_priority.remove(fea_j)
                        self.feature_mask_[np.where(self.input_features_ == fea_j)[0][0]] = False

        del self._column_priority

        if self._verbose:
            Logger().log('info', f'原始特征列数量：{self.get_feature_num_in()}')
            Logger().log('info', f'相关性低于{self._threshold}特征列数量：{self.get_feature_num_out()}')

        return self



