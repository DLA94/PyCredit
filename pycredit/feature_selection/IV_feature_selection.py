import pandas as pd
from pycredit.feature_selection._base import FeatureSelector
from typing import List
import numpy as np
from optbinning import BinningProcess
from optbinning.binning.binning_statistics import BinningTable
from pycredit.utils.log import Logger
from typing import Optional

class IVFeatureSelector(FeatureSelector):
    '''
    IV特征选择器，保留IV值在指定范围内的特征列。特征最优分箱结果通过 ``optbinning.BinningProcess`` 求解获得，

    IV反映的是一个变量对分类目标（如好/坏客户）的信息量贡献。计算方法如下：

    .. math::
        IV = \sum_{i} \left( P(\mathrm{good})_i - P(\mathrm{bad})_i \\right) \cdot \ln \left( \\frac{P(\mathrm{good})_i}{P(\mathrm{bad})_i} \\right)

    IV取值参考标准：

    .. list-table::
       :header-rows: 1
       :widths: 20 20 60
       :class: centered-table

       * - IV 区间
         - 区分度
         - 说明
       * - <0.02
         - 无预测能力
         - 变量无用，建议删除
       * - 0.02~0.1
         - 弱预测能力
         - 可用但作用小
       * - 0.1~0.3
         - 中等预测能力
         - 有参考价值
       * - 0.3~0.5
         - 强预测能力
         - 推荐保留
       * - >0.5
         - 非常强
         - 可能存在过拟合风险

    Parameters
    ----------
    max_n_prebins : int, default=20
        最大预分箱数目，超过此数目的特征将被忽略。
    min_prebin_size : float, default=0.05
        最小预分箱样本比例，低于此比例的特征将被忽略。
    iv_threshold : List, default=[0.02, None]
        IV值筛选阈值范围，格式为 [min, max]。如果max为None，则不限制最大IV值。
    n_jobs : int, default=1
        并行处理的工作线程数，-1表示使用所有可用核心。
    binning_process_params : dict, optional
        其他传递给 ``optbinning.BinningProcess`` 的参数。
    monotonic_trend : str, default='None'
        单调趋势，默认为自动检测升序或降序。支持的趋势包括 “auto”、“auto_heuristic” 和 “auto_asc_desc”，
        用于通过机器学习分类器自动确定最大化 IV 的趋势；“ascending”、“descending”、“concave”、“convex”、
        “peak” 和 “peak_heuristic” 用于允许出现峰值变化点；“valley” 和 “valley_heuristic” 用于允许出现
        谷值变化点。其中 “auto_heuristic”、“peak_heuristic” 和 “valley_heuristic” 采用启发式方法确定变
        化点，对于大规模样本具有显著更快的计算速度。
        参考 `optbinning.OptimalBinning <https://gnpalencia.org/optbinning/binning_binary.html#>`_ 文档了解更多细节。
    verbose : bool, default=False
        是否打印详细信息。

    Attributes
    ----------
    input_features\_ : np.ndarray
        输入特征列表。
    feature_mask\_ : List
        特征掩码, 选择特征列为True。
    score\_ : np.ndarray
        特征评分，表示每个特征的IV值。
    '''

    def __init__(self,
                 verbose: bool=False,
                 max_n_prebins: int=20,
                 min_prebin_size: float=0.05,
                 iv_threshold: List[float]=[0.02, 1e5],
                 n_jobs: int=1,
                 monotonic_trend: Optional[str]=None,
                 binning_process_params: Optional[dict]= None) -> None:
        super().__init__(verbose = verbose)

        self.__monotonic_trend = monotonic_trend

        self._binning_process = BinningProcess(
            variable_names=[],
            max_n_prebins = max_n_prebins,
            min_prebin_size = min_prebin_size,
            selection_criteria = {"iv": {"min": iv_threshold[0], "max": iv_threshold[1]}},
            verbose = verbose,
            n_jobs = n_jobs,
            **binning_process_params if binning_process_params is not None else {}
        )

    def fit(self, X, y, sample_weight=None) -> 'IVFeatureSelector':
        '''
        计算特征的IV值，并根据阈值范围筛选特征。

        Parameters
        ----------
        X : np.ndarray
            输入特征矩阵。
        y : np.ndarray
            目标变量。
        sample_weight : np.ndarray, optional
            样本权重，默认为None。

        Returns
        -------
        self : IVFeatureSelector
            返回自身实例。

        note
        ----
            输入数据特征列需为数值型！

        '''
        # 校验输入数据类型
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X需要是pandas.DataFrame类型')

        if not isinstance(y, (np.ndarray, pd.Series)):
            raise TypeError('y需要是numpy.ndarray或pandas.Series类型')

        # 校验列数据类型
        if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
            raise ValueError('X中的所有特征列必须是数值型')

        # 获取输入特征列表
        self.input_features_ = np.array(X.columns)

        # 最优分箱搜索
        self._binning_process.set_params(**{'variable_names': self.input_features_})

        ## 设置单调趋势
        if self.__monotonic_trend is not None:
            binning_fit_params = self._binning_process.get_params().get('binning_fit_params')
            if binning_fit_params is None:
                binning_fit_params = {feature:{"monotonic_trend": self.__monotonic_trend}
                                      for feature in self.input_features_}
            self._binning_process.set_params(**{'binning_fit_params': binning_fit_params})

        self._binning_process.fit(X, y, sample_weight=sample_weight)

        self.score_ = self._binning_process.summary()['iv'].astype(float).tolist()
        self.feature_mask_ = self._binning_process.get_support()

        if self._verbose:
            Logger().log('info', f'原始特征列数量：{self.get_feature_num_in()}')
            Logger().log('info', f'IV值符合要求特征列数量：{self.get_feature_num_out()}')

        return self

    def get_summary(self) -> pd.DataFrame:
        '''
        获取特征IV值的汇总信息。``verbose=True`` 时打印详细信息。

        Returns
        -------
        pd.DataFrame
            包含特征名、IV值和选择状态的DataFrame。
        '''
        summary = self._binning_process.summary()

        if self._verbose:
            Logger().log('info', f'特征分箱汇总信息如下：\n' +
                                 f' {self._binning_process.summary().to_markdown(index=False)}')
        return summary

    def _get_feature_binning_table(self, feature_name: str) -> BinningTable:
        '''
        获取指定特征的分箱表。

        Parameters
        ----------
        feature_name : str
            特征列名。

        Returns
        -------
        BinningTable
            分箱结果表。
        '''
        if feature_name not in self.input_features_:
            raise ValueError(f'特征 {feature_name} 不在输入特征列表中')

        binning_table = self._binning_process.get_binned_variable(feature_name).binning_table
        return binning_table

    def get_optimal_feature_binning(self, feature_name: str) -> pd.DataFrame:
        '''
        获取指定特征的分箱结果。

        Parameters
        ----------
        feature_name : str
            特征列名。

        Returns
        -------
        pd.DataFrame
            包含分箱结果的DataFrame。
        '''
        binning_table = self._get_feature_binning_table(feature_name).build()

        if self._verbose:
            Logger().log('info', f'特征 {feature_name} 的最优分箱结果如下：\n' +
                                 f' {binning_table.to_markdown(index=False)}')

        return binning_table

    def plot_optimal_feature_binning(self, feature_name: str) -> None:
        '''
        绘制指定特征的分箱结果图。

        Parameters
        ----------
        feature_name : str
            特征列名。

        Returns
        -------
        None
        '''
        binning_table = self._get_feature_binning_table(feature_name)
        binning_table.plot()

