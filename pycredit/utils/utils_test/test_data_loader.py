from sklearn.datasets import fetch_openml

def test_get_openml_dataset(id=43922):
    X, y = fetch_openml(data_id=id, as_frame=True, return_X_y=True)
    return X, y, X.columns


if __name__ == '__main__':
    X, y, attribute_names = test_get_openml_dataset()
    print(f'特征名称: {attribute_names}')
    print(f'X 形状: {X.shape}')
    print(f'y 形状: {y.shape}')

