# PyCredit

信用风险建模常用功能：

- **特征筛选（Featture Selection）**
  - NonUniqueFeatureSelector: 非唯一特征选择器，用于识别数据集中非唯一值的特征列
  - NonNullFeatureSelector: 缺失值特征选择器，用于识别数据集中缺失值比例超过阈值的特征列
  - CorrFeatureSelector: 相关性特征选择。基于特征之间的Person相关性进行筛选
  - IVFeatureSelector: IV特征选择器，保留IV值在指定范围内的特征列
  - FIFeatureSelector: 基于特征重要性的特征选择器，使用LightGBM进行特征重要性筛选
    
- **模型参数优化（Parameter Tuning）**
  - LGBMTuning: lightgbm分类器参数优化
    
- **模型评估（Metrics）**
  - model_eval: 评估模型的性能，包括准确率、AUC、F1、Precision、Recall、Kolmogorov-Smirnov (KS) 等
  - psi: PSI (Population Stability Index) 指标
  - model_eval_stat: 计算模型预测结果分层统计指标
  - plot_ks: 绘制KS曲线
  - plot_pr: 绘制PR曲线
  - plot_roc: 绘制ROC曲线

- **常用函数（Utils）**
  - DescriptiveStatistics: 描述性统计类，提供对数据的描述性统计方法
  - Logger: 日志类
  - to_polars_dataframe: 输入数据转换为 Polars DataFrame
  - score_transfer: 预测概率转评分
