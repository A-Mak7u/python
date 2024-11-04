# visualize_model_results.py

import pandas as pd
import matplotlib.pyplot as plt

# Загружаем результаты
results_df = pd.read_csv("model_results_extended.csv")

# Группируем результаты по моделям и создаем графики для каждой модели
models = results_df['Model'].unique()

for model in models:
    model_data = results_df[results_df['Model'] == model]

    plt.figure(figsize=(10, 6))
    plt.plot(model_data['Preprocessing'], model_data['R^2'], marker='o', label="R^2 Score")
    plt.plot(model_data['Preprocessing'], model_data['MSE'], marker='x', label="MSE")
    plt.title(f"Результаты для модели: {model}")
    plt.xlabel("Метод предобработки")
    plt.ylabel("Метрики")
    plt.legend()
    plt.grid(True)

    # Сохраняем график для каждой модели
    plt.savefig(f"{model}_results.png")
    plt.show()
