from pycredit.feature_selection._base import FeatureSelector
from pycredit.parameter_tuning.LGBM_tuning import LGBMTuning
import numpy as np
from pycredit.utils.log import Logger

class FIFeatureSelector(FeatureSelector):
    """
    基于特征重要性的特征选择器，使用LightGBM进行参数优化，选择重要性最高的特征作为输出。

    Parameters
    ----------
    verbose : bool, default=False
        是否打印日志信息。
    metric : str, default='KS'
        评估指标，可选'KS'、'AUC'、'ACC'。

    Attributes
    ----------
    input_features\_ : np.ndarray
        输入特征列表。
    feature_mask\_ : List
        特征掩码, 选择特征列为True。
    score\_ : np.ndarray
        特征评分，表示每个特征的IV值。
    best_estimator\_ : LightGBM.Booster
        最优模型。
    """

    def __init__(self, verbose=False, metric='KS'):
        super().__init__(verbose)

        self.lgbm_tuning = LGBMTuning(metric, verbose)

    def fit(self, X, y, cv=3, n_trials=100, n_jobs=-1):
        """
        计算特征重要性，选择重要性非0的特征作为输出。

        Parameters
        ----------
        X：array-like of shape (n_samples, n_features)
            训练数据集。
        y：array-like of shape (n_samples,)
            训练数据标签。
        cv：int, default=3
                交叉验证折数。
        n_trials: int, default=100
            优化参数的次数。
        n_jobs: int, default=-1
            并行计算的线程数。

        Returns
        -------
        self : FiFeatureSelector
            返回自身实例。
        """
        self.lgbm_tuning.optimize(X, y, cv=cv, n_trials=n_trials, n_jobs=n_jobs)
        self.best_estimator_ = self.lgbm_tuning.get_best_estimator()
        self.best_estimator_.fit(X, y)

        self.input_features_ = np.array(self.best_estimator_.feature_name_)
        self.score_ = np.array(self.best_estimator_.feature_importances_)
        self.feature_mask_ = (self.score_ > 0).tolist()

        if self._verbose:
            Logger().log('info', f'原始特征列数量：{self.get_feature_num_in()}')
            Logger().log('info', f'IV值符合要求特征列数量：{self.get_feature_num_out()}')

        return self