# temperature_pipeline.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Игнорируем предупреждения для более чистого вывода
warnings.filterwarnings("ignore")


# 1. Загрузка данных
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset


# 2. Предобработка данных
def preprocess_data(data, method):
    features = data.drop(columns=['T_rp5'])
    target = data['T_rp5']

    if method == "None":
        return features, target
    elif method == "StandardScaler":
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)
    elif method == "PolynomialFeatures":
        poly = PolynomialFeatures(degree=2, include_bias=False)
        features = poly.fit_transform(features)

    return features, target


# 3. Оценка модели
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# 4. Основная функция для тестирования комбинаций предобработок и моделей
def main():
    file_path = 'LST_final_TRUE.csv'  # Замените на путь к вашему файлу
    data = load_data(file_path)

    # Параметры для тестирования
    preprocessing_methods = ["None", "StandardScaler", "MinMaxScaler", "PolynomialFeatures"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=42)
    }

    # Таблица для хранения результатов
    results = []

    for preproc in preprocessing_methods:
        X, y = preprocess_data(data, preproc)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            mse, r2 = evaluate_model(model, X_test, y_test)
            results.append({
                "Preprocessing": preproc,
                "Model": model_name,
                "MSE": mse,
                "R^2": r2
            })

    # Создаем DataFrame с результатами
    results_df = pd.DataFrame(results)
    print(results_df)

    # Сохраняем результаты в CSV-файл для удобства
    results_df.to_csv("model_results.csv", index=False)


if __name__ == "__main__":
    main()
