import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
dataset = pd.read_csv('LST_final_TRUE.csv')

# Информация о датасете
print("Информация о датасете:")
print(dataset.info())

# Первые несколько строк данных
print("\nПервые строки данных:")
print(dataset.head())

# 1. Проверка на пропуски
print("\nПроверка на пропуски:")
print(dataset.isnull().sum())

# 2. Статистические характеристики
print("\nСтатистические характеристики данных:")
pd.set_option('display.max_columns', None)
print(dataset.describe())

# 3. Проверка типов данных
print("\nТипы данных в столбцах:")
print(dataset.dtypes)

# 4. Анализ распределения данных (гистограммы для числовых признаков)
print("\nРаспределение данных для числовых признаков:")
dataset.hist(figsize=(12, 10), bins=50)
plt.tight_layout()
plt.show()

# 5. Выявление выбросов (boxplot для каждого числового признака)
print("\nАнализ выбросов с помощью boxplot:")
numerical_columns = dataset.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(dataset[col])
    plt.title(col)
plt.tight_layout()
plt.show()
