import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# 1 loading and checking
def load_and_inspect_data(file_path):
    dataset = pd.read_csv(file_path)
    print("данные:")
    print(dataset.info())
    print("первые строки данных:")
    print(dataset.head())
    return dataset


# 2 emty spaces
def handle_missing_values(dataset):
    print("\nпроверка наличия пропусков:")
    print(dataset.isnull().sum())
    dataset.fillna(dataset.median(), inplace=True)
    return dataset


# 3
def scale_features(dataset):
    features = dataset.drop(columns=['T_rp5'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target = dataset['T_rp5']
    return features_scaled, target


# 4
def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)


# 5 подбор гиперпараметров и обучение
def train_model_with_hyperparameter_tuning(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)

    '''
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    '''

    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, None],
        'min_samples_split': [2, 5]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
    grid_search.fit(X_train, y_train)

    print(f"\nЛучшие параметры: {grid_search.best_params_}")
    print(f"Лучшее значение R^2 на обучении: {grid_search.best_score_}")

    return grid_search.best_estimator_


# 6 final test
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'\nMean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
    return y_test, y_pred


# 7 visual
def plot_results(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Настоящие значения')
    plt.plot(y_pred, label='Предсказанные значения', alpha=0.7)
    plt.legend()
    plt.title('Сравнение реальных и предсказанных значений температуры')
    plt.show()


def main():
    file_path = 'LST_final_TRUE.csv'  # Замените на путь к вашему файлу
    dataset = load_and_inspect_data(file_path)
    dataset = handle_missing_values(dataset)
    features, target = scale_features(dataset)
    X_train, X_test, y_train, y_test = split_data(features, target)
    model = train_model_with_hyperparameter_tuning(X_train, y_train)
    y_test, y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
