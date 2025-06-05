from sklearn import metrics
import pandas as pd
import numpy as np

def model_eval(y_true, y_pred, eval_metric=None, threshold=0.5):
    '''
    评估模型的性能，包括准确率、AUC、F1、Precision、Recall、Kolmogorov-Smirnov (KS) 等。

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        groud-truth标签
    y_pred: array-like of shape (n_samples,)
        预测标签, 预测为1的概率
    eval_metric: list of str, default=None
        评估指标，包括：

        - 'ACC': Accuracy

        - 'AUC': Area Under the Receiver Operating Characteristic Curve (ROC AUC)

        - 'F1': F1 score

        - 'Precision': Precision

        - 'Recall': Recall

        - 'KS': Kolmogorov-Smirnov statistic
    threshold: float, default=0.5
        预测为1的概率阈值

    Returns
    -------
    eval_results: pd.DataFrame
        评估结果
    '''
    y_pred_ = np.array(y_pred) > threshold

    if eval_metric is None:
        eval_metric = ['ACC', 'AUC', 'F1', 'Precision', 'Recall', 'KS']

    eval_results = []
    for metric in eval_metric:
        if metric == 'AUC':
            eval_results.append(metrics.roc_auc_score(y_true, y_pred))
        elif metric == 'F1':
            eval_results.append(metrics.f1_score(y_true, y_pred_))
        elif metric == 'Precision':
            eval_results.append(metrics.precision_score(y_true, y_pred_))
        elif metric == 'Recall':
            eval_results.append(metrics.recall_score(y_true, y_pred_))
        elif metric == 'KS':
            fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
            eval_results.append(max(abs(tpr - fpr)))
        elif metric == 'ACC':
            eval_results.append(metrics.accuracy_score(y_true, y_pred_))
        else:
            raise ValueError(f"Invalid evaluation metric: {metric}")

    eval_results = pd.DataFrame({'Value': eval_results}, index=eval_metric)
    return eval_results


def psi(y_expect, y_actual, bins=None):
    '''
    计算 PSI (Population Stability Index) 指标。

    Parameters
    ----------
    y_expect: array-like of shape (n_samples,)
        期望标签
    y_actual: array-like of shape (n_samples,)
        实际标签
    bins: List
        分箱，默认为 None

    Returns
    -------
    psi_value: float
        PSI 值
    '''
    if bins is None:
        bins = [-1e-5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    expected_per = pd.cut(y_expect, bins=bins, labels=False)
    expected_per = np.bincount(expected_per) / len(expected_per)

    actual_per = pd.cut(y_actual, bins=bins, labels=False)
    actual_per = np.bincount(actual_per) / len(actual_per)

    psi_value = np.sum((actual_per - expected_per) * np.log(actual_per / expected_per))
    return psi_value


def model_eval_stat(y_true, y_pred, bins=None):
    '''
    计算模型统计指标

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        groud-truth标签
    y_pred: array-like of shape (n_samples,)
        预测标签, 预测为1的概率
    bins: List
        分箱，默认为 None

    Returns
    -------
    stat_res: pd.DataFrame
        模型统计指标
    '''

    if bins is None:
        bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    df = pd.DataFrame({'y_true': y_true, 'y_pred': -y_pred})
    df['bin'] = pd.qcut(df['y_pred'], q=bins, labels=False)
    df = df.groupby('bin').agg({
        'y_true': [lambda x: len(x)-sum(x), lambda x: sum(x)],
        'y_pred': [lambda x: min(-x), lambda x: max(-x)]
    }).reset_index()

    df.columns = ['Bin_Order', 'Negative', 'Positive', 'y_pred_min', 'y_pred_max']
    df['Prob'] = ['({},{}]'.format(round(left, 4), round(right, 4)) for left, right in
                  zip(df['y_pred_min'][:-1].tolist() + [0],
                      [1] + df['y_pred_max'][1:].tolist())]
    df['Bin_Count'] = df['Negative'] + df['Positive']
    df['Negative_Margin_Percentage'] = round(df['Negative'] / df['Bin_Count'] * 100, 2).apply(lambda x: str(x) + '%')
    df['Positive_Margin_Percentage'] = round(df['Positive'] / df['Bin_Count'] * 100, 2).apply(lambda x: str(x) + '%')

    df['Positive_Cumsum'] = df['Positive'].cumsum()
    df['Positive_Cumsum_Percentage'] = (round(df['Positive_Cumsum'] / df['Positive'].sum() * 100, 2).
                                        apply(lambda x: str(x) + '%'))
    df['Total_Percentage'] = round(df['Bin_Count'].cumsum() / df['Bin_Count'].sum() * 100, 2).apply(lambda x: str(x) + '%')
    df['Lift'] = round((df['Positive_Cumsum'] / df['Positive'].sum()) / (df['Bin_Count'].cumsum() / df['Bin_Count'].sum()), 4)

    stat_res = df[[
        'Bin_Order', 'Prob', 'Bin_Count', 'Total_Percentage', 'Negative', 'Negative_Margin_Percentage',
        'Positive', 'Positive_Margin_Percentage', 'Positive_Cumsum', 'Positive_Cumsum_Percentage', 'Lift'
    ]]

    return stat_res
