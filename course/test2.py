import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

# 3 macshtab
def scale_features(dataset):
    features = dataset.drop(columns=['T_rp5'])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target = dataset['T_rp5']
    return features_scaled, target

# 4
def split_data(features, target):
    return train_test_split(features, target, test_size=0.2, random_state=42)

# 5 learning
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# 6 test
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
    model = train_model(X_train, y_train)
    y_test, y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)

if __name__ == "__main__":
    main()
