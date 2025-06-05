if __name__ == '__main__':
    from sklearn.datasets import fetch_openml, load_breast_cancer
    from pycredit.feature_selection.Feature_importance_feature_selection import FIFeatureSelector

    X, y = fetch_openml(data_id=23512, as_frame=True, return_X_y=True)
    # X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    y = y.astype(int)

    fs = FIFeatureSelector(verbose=False, metric='KS')
    fs.fit(X, y, cv=3, n_trials=20, n_jobs=-1)

    print(fs.get_feature_names_in())
    print(fs.get_feature_names_out())
    print(fs.get_feature_num_in())
    print(fs.get_feature_num_out())
    print(fs.get_score())