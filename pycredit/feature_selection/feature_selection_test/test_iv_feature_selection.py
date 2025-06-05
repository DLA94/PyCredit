if __name__ == '__main__':
    from pycredit.utils.utils_test.test_data_loader import test_get_openml_dataset
    from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
    from pycredit.feature_selection.IV_feature_selection import IVFeatureSelector
    import pandas as pd
    import warnings

    warnings.filterwarnings('ignore')

    X, y, _ = test_get_openml_dataset()
    X = X.apply(lambda col: col.astype(str))
    X = X.replace('?', None)
    X = pd.DataFrame(OrdinalEncoder().fit_transform(X), columns=X.columns)
    y = LabelEncoder().fit_transform(y)

    print(X.head())
    print(y)

    fs = IVFeatureSelector(verbose=True, iv_threshold=[0.02, 0.5], n_jobs=-1,)
    fs.fit(X, y)

    print(fs.get_feature_names_in())
    print(fs.get_feature_names_out())
    print(fs.get_feature_num_in())
    print(fs.get_feature_num_out())
    print(fs.get_score())
    fs.get_summary()
    fs.get_optimal_feature_binning('veil_color')
    fs.plot_optimal_feature_binning('veil_color')