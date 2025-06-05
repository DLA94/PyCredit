import matplotlib.pyplot as plt
from sklearn import metrics
from typing import List

def plot_ks(y_true, y_pred):
    '''
    绘制KS曲线

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        实际标签
    y_pred: array-like of shape (n_samples,)
        预测标签

    Returns
    -------
    None
    '''

    plt.figure(figsize=(8, 6))
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ks = max(abs(tpr - fpr))
    plt.plot(thresholds, abs(tpr - fpr), 'k', label=f'KS Curve (ks = {ks:.3f})')
    plt.plot(thresholds, tpr, 'r--', label='TPR')
    plt.plot(thresholds, fpr, 'g--', label='FPR')
    plt.xlim([0, 1])
    plt.ylim([-0.01, 1.1])
    plt.xlabel('Thresholds')
    plt.ylabel('TPR / FPR')
    plt.grid(True)
    plt.gca().invert_xaxis()
    plt.legend()
    plt.show()

def plot_pr(y_true, y_pred):
    '''
    绘制PR曲线

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        实际标签
    y_pred: array-like of shape (n_samples,)
        预测标签

    Returns
    -------
    None
    '''
    plt.figure(figsize=(8, 6))
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    ap = metrics.average_precision_score(y_true, y_pred)
    plt.plot(recall, precision, 'k', label=f'PR Curve (AP = {ap:.3f})')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_roc(y_true, y_preds: List, model_names: List):
    '''
    绘制ROC曲线

    Parameters
    ----------
    y_true: array-like of shape (n_samples,)
        实际标签
    y_preds: List of array-like of shape (m_models, n_samples)
        预测标签
    model_names: List of str (m_models,)
        模型名称

    Returns
    -------
    None
    '''
    plt.figure(figsize=(8, 6))
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_names[i]} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid(True)
    plt.legend()
    plt.show()