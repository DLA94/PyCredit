import lightgbm as lgb
import optuna
from functools import partial
from pycredit.metrics.model_eval import model_eval
from sklearn.model_selection import KFold
import numpy as np
from pycredit.utils.utils_func import to_polars_dataframe
from optuna import visualization

optuna.logging.set_verbosity(optuna.logging.WARNING)

class LGBMTuning():
    """
    lightgbm分类器参数优化

    Parameters
    ----------
    metric : str
        评估指标，支持ACC，AUC，KS
    threshold : float
        预测阈值，默认0.5

    Attributes
    ----------
    study\_ : optuna.study.Study
        优化过程记录。
    base_estimator\_ : lightgbm.LGBMClassifier
        基分类器。

    """
    def __init__(self,
                 metric: str = 'KS',
                 threshold: float = 0.5
                 ):
        sampler = optuna.samplers.TPESampler(seed=42)
        self.study_ = optuna.create_study(direction='maximize', sampler=sampler)

        if metric not in ['ACC', 'AUC', 'KS']:
            raise ValueError('metric should be ACC, AUC or KS')
        else:
            self.metric_ = partial(model_eval, eval_metric=[metric], threshold=threshold)

        self.base_estimator_ = lgb.LGBMClassifier

    def objective(self, trial):
        '''
        定义目标函数

        Parameters
        ----------
        trial: optuna.trial.Trial
            优化过程中的实验对象

        Returns
        -------
        float
            目标函数值
        '''
        # 定义分类器参数搜索空间
        # ensemble params
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)      # 树的数量
        learning_rate = trial.suggest_float('learning_rate', 5e-5, 0.1, log=True)    # 学习率
        reg_alpha = trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True)          # L1正则化项
        reg_lambda = trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True)         # L2正则化项

        # tree params
        boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss'])  # 树类型
        extra_trees = trial.suggest_categorical('extra_trees', [True, False])                 # 是否使用extra tree
        max_depth = trial.suggest_int('max_depth', 2, 10)                                     # 树的最大深度
        num_leaves = trial.suggest_int('num_leaves', 2, 2**max_depth)                                    # 叶子节点的数量
        min_child_samples = trial.suggest_int('min_child_samples',
                                              int(self._train_size * 0.01),
                                              int(self._train_size * 0.1))                        # 叶子节点最小样本数
        min_split_gain = trial.suggest_float('min_split_gain', 0.0, 20.0)         # 分裂增益最小值

        drop_rate = trial.suggest_float('drop_rate', 0.005, 0.3)                 # dropout率
        top_rate = trial.suggest_float('top_rate', 0.1, 0.5)                         # 过采样率
        other_rate = trial.suggest_float('other_rate', 0.1, 0.5)                     # 其他采样率
        subsample_freq = trial.suggest_int('subsample_freq', 1, 5)                     # 子采样频率
        subsample = trial.suggest_float('subsample', 0.3, 1.0)                         # 子采样比例

        max_bin = trial.suggest_int('max_bin', 5, 255)                               # 最大分箱数
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 0.95)           # 列采样比例
        colsample_bynode = trial.suggest_float('colsample_bynode', 0.5, 1)             # 节点采样比例

        random_state = trial.suggest_int('random_state', 1, 1e10)                     # 随机种子

        # 参数字典
        params = {
            'objective': 'binary', 'importance_type': 'gain', 'silent': False, 'verbosity':-1,
            'n_estimators': n_estimators, 'learning_rate': learning_rate,
            'reg_alpha': reg_alpha,'reg_lambda': reg_lambda,
            'boosting_type': boosting_type, 'extra_trees': extra_trees,
            'max_depth': max_depth, 'num_leaves': num_leaves,
            'min_child_samples': min_child_samples,'min_split_gain': min_split_gain,
            'drop_rate': drop_rate, 'top_rate': top_rate, 'other_rate': other_rate,
            'subsample_freq': subsample_freq,'subsample': subsample,
            'max_bin': max_bin, 'colsample_bytree': colsample_bytree,
            'colsample_bynode': colsample_bynode, 'random_state': random_state
        }

        # 剔除无效参数
        params = {
            key: value for key, value in params.items()
            if key not in {
                'gbdt': ['drop_rate', 'top_rate', 'other_rate'],
                'dart': ['drop_rate', 'top_rate'],
                'goss': ['other_rate', 'subsample_freq', 'subsample']
            }[boosting_type]
        }

        score = []
        for train_index, test_index in self._kfold.split(self._X, self._y):
            X_train, X_test = self._X.iloc[train_index], self._X.iloc[test_index]
            y_train, y_test = self._y[train_index], self._y[test_index]

            estimator = self.base_estimator_(**params)
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict_proba(X_test)[:,1]
            score.append(self.metric_(y_test, y_pred).values[0][0])
        return np.mean(score)

    def optimize(self, X, y, cv=3, n_trials=100, n_jobs=-1):
        '''
        优化参数

        Parameters
        ----------
        X: pandas.DataFrame
            训练集特征
        y: pandas.Series or numpy.ndarray or list
            训练集标签
        cv: int
            交叉验证折数
        n_trials: int
            优化次数
        n_jobs: int
            并行数

        Returns
        -------
        self: LGBTunning
            优化后的实例
        '''

        self._train_size = int(len(y) / cv * (cv - 1))
        self._kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

        self._X = to_polars_dataframe(X).to_pandas()
        if hasattr(y, 'to_numpy'):
            self._y = y.to_numpy()
        elif isinstance(y, (np.ndarray, list)):
            self._y = np.array(y)
        else:
            raise ValueError('y should be pandas.Series or numpy.ndarray or list')

        self.study_.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

        del self._X, self._y, self._kfold, self._train_size

        return self

    def get_best_estimator(self):
        '''
        获取最优分类器

        Returns
        -------
        object
            最优分类器
        '''
        params = self.get_best_params()
        best_estimator = self.base_estimator_(**params)
        return best_estimator

    def get_best_params(self):
        '''
        获取最优参数

        Returns
        -------
        best_params: dict
            最优参数字典
        '''
        params = {'objective': 'binary', 'importance_type': 'gain', 'silent': False, 'verbosity': -1}
        params = {**params, **self.study_.best_params}
        params = {
            key: value for key, value in params.items()
            if key not in {
                'gbdt': ['drop_rate', 'top_rate', 'other_rate'],
                'dart': ['drop_rate', 'top_rate'],
                'goss': ['other_rate', 'subsample_freq', 'subsample']
            }[params['boosting_type']]
        }
        return params

    def study_plot(self, info='history'):
        '''
        可视化优化过程

        Parameters
        ----------
        info: str
            可视化信息，包括history，parameter_correlation，timeline，parameter_importance

        Returns
        -------
        None
        '''

        if info == 'history':
            visualization.plot_optimization_history(self.study_).show()
        elif info == 'parameter_correlation':
            visualization.plot_parallel_coordinate(self.study_).show()
        elif info == 'timeline':
            visualization.plot_timeline(self.study_).show()
        elif info == 'parameter_importance':
            visualization.plot_param_importances(self.study_).show()
        else:
            raise ValueError('info should be history, parameter_correlation, timeline or parameter_importance')