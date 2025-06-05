if __name__ == '__main__':
    from pycredit.utils.utils_test.test_data_loader import test_get_openml_dataset
    from sklearn.preprocessing import OrdinalEncoder
    from pycredit.feature_selection.Corr_feature_selection import CorrFeatureSelector
    import pandas as pd
    import warnings

    warnings.filterwarnings('ignore')

    X, y, _ = test_get_openml_dataset()
    X = X.apply(lambda col: col.astype(str))
    X = X.replace('?', None)
    X = pd.DataFrame(OrdinalEncoder().fit_transform(X), columns=X.columns)
    X = X.drop('veil_type', axis=1)

    fs = CorrFeatureSelector(heuristic_mode='local', threshold=0.5, verbose=True)
    fs.fit(X)
    print(fs.get_feature_names_out())

    fs = CorrFeatureSelector(heuristic_mode='global', threshold=0.5, verbose=True)
    fs.fit(X)
    print(fs.get_feature_names_out())