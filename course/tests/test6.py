import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

# 2 empty spaces
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

# 5 hyperparameter tuning and training
def train_model_with_hyperparameter_tuning(X_train, y_train):
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)  # Использование всех ядер процессора

    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions,
                                       n_iter=100, cv=5, n_jobs=-1, scoring='r2', random_state=42)
    random_search.fit(X_train, y_train)

    print(f"\nЛучшие параметры: {random_search.best_params_}")
    print(f"Лучшее значение R^2 на обучении: {random_search.best_score_}")

    return random_search.best_estimator_

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
    file_path = 'LST_final_TRUE.csv'
    dataset = load_and_inspect_data(file_path)
    dataset = handle_missing_values(dataset)
    features, target = scale_features(dataset)
    X_train, X_test, y_train, y_test = split_data(features, target)
    model = train_model_with_hyperparameter_tuning(X_train, y_train)
    y_test, y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()

'''
данные:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19077 entries, 0 to 19076
Data columns (total 12 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   H                          19077 non-null  int64  
 1   TWI                        19077 non-null  float64
 2   Aspect                     19077 non-null  float64
 3   Hillshade                  19077 non-null  int64  
 4   Roughness                  19077 non-null  int64  
 5   Slope                      19077 non-null  float64
 6   Temperature_merra_1000hpa  19077 non-null  float64
 7   T_rp5                      19077 non-null  float64
 8   Time                       19077 non-null  int64  
 9   DayOfYear                  19077 non-null  int64  
 10  X                          19077 non-null  float64
 11  Y                          19077 non-null  float64
dtypes: float64(7), int64(5)
memory usage: 1.7 MB
None
первые строки данных:
     H   TWI  Aspect  Hillshade  Roughness  ...  T_rp5  Time  DayOfYear     X     Y
0  110  6.57     0.0        184          2  ...   14.1     0        152  50.8  42.0
1  110  6.57     0.0        184          2  ...   20.1   720        152  50.8  42.0
2  110  6.57     0.0        184          2  ...   12.3   360        152  50.8  42.0
3  110  6.57     0.0        184          2  ...   21.5  1080        152  50.8  42.0
4  110  6.57     0.0        184          2  ...   24.1   720        153  50.8  42.0

[5 rows x 12 columns]

проверка наличия пропусков:
H                            0
TWI                          0
Aspect                       0
Hillshade                    0
Roughness                    0
Slope                        0
Temperature_merra_1000hpa    0
T_rp5                        0
Time                         0
DayOfYear                    0
X                            0
Y                            0
dtype: int64
C:\Users\Tom\PycharmProjects\python\.venv\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
  _data = np.array(data, dtype=dtype, copy=copy,

Лучшие параметры: {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 20}
Лучшее значение R^2 на обучении: 0.8306926856537892

Mean Squared Error: 6.203933694614795
R^2 Score: 0.8354724739788113

выполнялся где то 5 минут
'''