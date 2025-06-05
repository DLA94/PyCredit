if __name__ == '__main__':
    from pycredit.utils.utils_test.test_data_loader import test_get_openml_dataset
    from pycredit.feature_selection.non_unique_feature_selection import NonUniqueFeatureSelector

    X, y, _ = test_get_openml_dataset()

    fs = NonUniqueFeatureSelector(verbose=True)
    fs.fit(X)
    X_ = fs.transform(X)
    print(X_.shape)

    X_ = fs.fit_transform(X)
    print(X_.shape)

    print(fs.get_feature_names_in())
    print(fs.get_feature_names_out())
    print(fs.get_feature_num_in())
    print(fs.get_feature_num_out())
    print(fs.get_score())
