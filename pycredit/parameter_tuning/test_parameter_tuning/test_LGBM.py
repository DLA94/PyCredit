if __name__ == '__main__':
    from sklearn.datasets import fetch_openml, load_breast_cancer
    from pycredit.parameter_tuning.LGBM_tuning import LGBMTuning
    import lightgbm as lgb
    from pycredit.metrics.model_eval import model_eval
    # X, y = fetch_openml(data_id=23512, as_frame=True, return_X_y=True)
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    y = y.astype(int)

    tunning = LGBMTuning(metric='AUC')
    tunning.optimize(X, y, cv=3, n_trials=20)

    params = tunning.study_.best_params
    params = {
        key: value for key, value in params.items()
        if key not in {
            'gbdt': ['drop_rate', 'top_rate', 'other_rate'],
            'dart': ['drop_rate', 'top_rate'],
            'goss': ['other_rate', 'subsample_freq', 'subsample']
        }[params['boosting_type']]
    }

    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # clf = lgb.LGBMClassifier(**params)
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict_proba(X_test)[:,1]
    # print(model_eval(y_test, y_pred, eval_metric=['KS']))

    tunning.study_plot('history')
    tunning.study_plot('parameter_correlation')
    tunning.study_plot('timeline')
    tunning.study_plot('parameter_importance')