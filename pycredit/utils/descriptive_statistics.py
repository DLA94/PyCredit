import polars as pl

from pycredit.utils.log import Logger
from pycredit.utils.utils_func import to_polars_dataframe

class DescriptiveStatistics:
    '''
    描述性统计类，提供对数据的描述性统计方法
    '''

    @staticmethod
    def column_types(data):
        '''
        获取数据的列类型

        Parameters
        ----------
        data: array_like or DataFrame
            待获取列类型的数据

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的名称和类型
        '''
        data = to_polars_dataframe(data)
        stat_res = [{'feature': k, 'type': str(v)} for k, v in data.schema.items()]
        stat_res = pl.DataFrame(stat_res)
        return stat_res

    @staticmethod
    def null_count(data, null_value=[]):
        '''
        计算数据中每列的缺失值数量

        Parameters
        ----------
        data: array_like or DataFrame
            待计算缺失值数量的数据
        null_value: list, optional
            可选参数，指定哪些值被视为缺失值，默认为空列表

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的缺失值数量
        '''
        data = to_polars_dataframe(data)

        # 统计每列值为null或在null_value列表中的数量
        stat_res = data.select(
            [(pl.col(col).is_null() | pl.col(col).is_in(null_value)).sum().alias(col) for col in data.columns]
        ).transpose(include_header=True)\
         .rename({'column': 'feature', 'column_0': 'null_count'})
        return stat_res

    @staticmethod
    def min(data):
        '''
        计算数据的最小值

        Parameters
        ----------
        data: array_like or DataFrame
            待计算最小值的数据

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的最小值
        '''
        stat_res = to_polars_dataframe(data).min()
        stat_res = stat_res.transpose(include_header=True)\
                           .rename({'column': 'feature', 'column_0': 'min'})
        return stat_res

    @staticmethod
    def max(data):
        '''
        计算数据的最大值

        Parameters
        ----------
        data: array_like or DataFrame
            待计算最大值的数据

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的最大值
        '''
        stat_res = to_polars_dataframe(data).max()
        stat_res = stat_res.transpose(include_header=True)\
                           .rename({'column': 'feature', 'column_0': 'max'})
        return stat_res

    @staticmethod
    def mean(data):
        '''
        计算数据的平均值

        Parameters
        ----------
        data: array_like or DataFrame
            待计算平均值的数据

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的平均值
        '''
        stat_res = to_polars_dataframe(data).mean()
        stat_res = stat_res.transpose(include_header=True)\
                           .rename({'column': 'feature', 'column_0': 'mean'})
        return stat_res

    @staticmethod
    def median(data):
        '''
        计算数据的中位数

        Parameters
        ----------
        data: array_like or DataFrame
            待计算中位数的数据

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的中位数
        '''
        stat_res = to_polars_dataframe(data).median()
        stat_res = stat_res.transpose(include_header=True)\
                           .rename({'column': 'feature', 'column_0': 'median'})
        return stat_res

    @staticmethod
    def percentile(data, q: int=0.5):
        '''
        计算数据的百分位数

        Parameters
        ----------
        data: array_like or DataFrame
            待计算百分位数的数据
        q: int, default=0.5
            百分位数，默认值为 0.5

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的百分位数
        '''
        stat_res = to_polars_dataframe(data).quantile(q)
        stat_res = stat_res.transpose(include_header=True)\
                           .rename({'column': 'feature', 'column_0': f'{int(q*100)}%'})
        return stat_res

    @staticmethod
    def statistics(data,
                   stat=['type', 'null_count', 'min', 'max', 'mean', 'median'],
                   q=[0.25, 0.5, 0.75],
                   null_list=[]):
        '''
        计算数据的描述性统计量

        Parameters
        ----------
        data: array_like or DataFrame
            待统计数据
        stat: list, default=['type', 'null_count', 'min', 'max', 'mean', 'median']
            需要计算的统计量列表，默认包括 'type', 'null_count', 'min', 'max', 'mean', 'median'
        q: list, default=[0.25, 0.5, 0.75]
            百分位数列表，默认包括 0.25, 0.5, 0.75
        null_list: list, default=[]
            可选参数，指定哪些值被视为缺失值，默认为空列表

        Returns
        -------
        stat_res: pl.DataFrame
            包含每列的统计量
        '''
        data = to_polars_dataframe(data)
        Logger().log('info', f'输入数据包含{data.shape[0]}行{data.shape[1]}列')

        stat_res = pl.DataFrame({'feature': data.columns})

        # 计算每个统计量并添加到结果中
        if 'type' in stat:
            stat_res = stat_res.join(DescriptiveStatistics.column_types(data), on='feature')

        if 'null_count' in stat:
            stat_res = stat_res.join(DescriptiveStatistics.null_count(data, null_value=null_list), on='feature')

        if 'min'  in stat:
            stat_res = stat_res.join(DescriptiveStatistics.min(data), on='feature')

        if 'max' in stat:
            stat_res = stat_res.join(DescriptiveStatistics.max(data), on='feature')

        if 'mean' in stat:
            stat_res = stat_res.join(DescriptiveStatistics.mean(data), on='feature')

        if 'median' in stat:
            stat_res = stat_res.join(DescriptiveStatistics.median(data), on='feature')

        for q_value in q:
            if isinstance(q_value, (int, float)):
                stat_res = stat_res.join(DescriptiveStatistics.percentile(data, q=q_value), on='feature')
            else:
                raise ValueError(f"Invalid percentile value: {q_value}. Must be a number.")
        return stat_res

    @staticmethod
    def corr(data):
        '''
        计算数据的相关系数矩阵

        Parameters
        ----------
        data: array_like or DataFrame
            待计算相关系数的数据

        Returns
        -------
        corr_res: pl.DataFrame
            包含相关系数矩阵
        '''
        data = to_polars_dataframe(data)
        corr_res = data.corr()
        return corr_res
