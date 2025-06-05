if __name__ == '__main__':

    from pycredit.utils.utils_test.test_data_loader import test_get_openml_dataset
    from sklearn.preprocessing import OrdinalEncoder
    from pycredit.utils.descriptive_statistics import DescriptiveStatistics

    X, y, _ = test_get_openml_dataset()
    X = OrdinalEncoder().fit_transform(X)

    des_res = DescriptiveStatistics.statistics(X)
    print(des_res)

    corr_res = DescriptiveStatistics.corr(X)
    print(corr_res)