import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np


def load_and_inspect_data(file_path):
    dataset = pd.read_csv(file_path)
    print("Информация о данных:")
    print(dataset.info())
    print("Первые строки данных:")
    print(dataset.head())
    return dataset


def handle_missing_values(dataset):
    print("\nПроверка наличия пропусков:")
    print(dataset.isnull().sum())
    dataset.fillna(dataset.median(), inplace=True)
    return dataset


def scale_features(dataset):
    features = dataset.drop(columns=['T_rp5'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target = dataset['T_rp5']
    return features_scaled, target


def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)


def train_model_with_hyperparameter_tuning(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    param_distributions = {
        'n_estimators': [100, 150, 200, 250],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']  # 'auto' убрано
    }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                       n_iter=200, cv=10, n_jobs=-1, scoring='r2', random_state=42)
    random_search.fit(X_train, y_train)

    print(f"\nЛучшие параметры: {random_search.best_params_}")
    print(f"Лучшее значение R^2 на обучении: {random_search.best_score_}")

    return random_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return y_test, y_pred


def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Настоящие значения')
    plt.plot(y_pred, label='Предсказанные значения', alpha=0.7)
    plt.legend()
    plt.title('Сравнение реальных и предсказанных значений температуры')
    plt.show()


def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title('Важность признаков')
    plt.bar(range(features.shape[1]), importances[indices], align='center')
    plt.xticks(range(features.shape[1]), [features.columns[i] for i in indices], rotation=90)
    plt.xlim([-1, features.shape[1]])
    plt.show()


def main():
    file_path = 'LST_final_TRUE.csv'
    dataset = load_and_inspect_data(file_path)
    dataset = handle_missing_values(dataset)
    features, target = scale_features(dataset)
    X_train, X_test, y_train, y_test = split_data(features, target)
    model = train_model_with_hyperparameter_tuning(X_train, y_train)
    y_test, y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)
    plot_feature_importance(model, dataset.drop(columns=['T_rp5']))


if __name__ == "__main__":
    main()

'''
выстраивал очень долго, почти 10 минут

C:\Users\Tom\PycharmProjects\python\.venv\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,

Лучшие параметры: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 20}
Лучшее значение R^2 на обучении: 0.836804138345699

Mean Squared Error: 6.203933694614795
R^2 Score: 0.8354724739788113
'''