import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных (замените 'data.csv' на путь к вашим данным)
df = pd.read_csv('LST_final_TRUE.csv')  # если данные находятся в CSV файле
# Замените это на ваш способ загрузки данных
# Пример:
# df = pd.DataFrame(...) # создайте DataFrame из ваших данных

# Предположим, что ваши данные загружены в DataFrame df
# Разделите данные на признаки (X) и целевую переменную (y)
X = df.drop('Temperature_merra_1000hpa', axis=1)  # Убедитесь, что 'Temperature_merra_1000hpa' - это ваша целевая переменная
y = df['Temperature_merra_1000hpa']

# Определение модели
model = RandomForestRegressor()

# Определение параметров для RandomizedSearchCV
param_distributions = {
    'n_estimators': [200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
}

# Инициализация RandomizedSearchCV
search = RandomizedSearchCV(model, param_distributions, n_iter=100, cv=5, n_jobs=-1, random_state=42)

# Обучение модели
search.fit(X, y)

# Предсказания
y_pred = search.predict(X)

# Оценка модели
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Вывод результатов
print("Лучшие параметры:", search.best_params_)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Дополнительно: вывод важности признаков
feature_importances = search.best_estimator_.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Важность признаков:\n", importance_df)

'''
C:\Users\Tom\PycharmProjects\python\.venv\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,
Лучшие параметры: {'n_estimators': 200, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'log2', 'max_depth': None}
Mean Squared Error: 3.410107297821423
R^2 Score: 0.9077466146773243
Важность признаков:
       Feature  Importance
8   DayOfYear    0.296702
6       T_rp5    0.172117
9           X    0.099764
1         TWI    0.087290
10          Y    0.070646
0           H    0.056025
3   Hillshade    0.050499
7        Time    0.048678
5       Slope    0.044839
2      Aspect    0.043268
4   Roughness    0.030171

Process finished with exit code 0
'''