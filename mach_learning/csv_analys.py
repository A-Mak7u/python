import pandas as pd

file_path = "task19.csv"

data = pd.read_csv(file_path, encoding='windows-1251', sep=';')

# преобразования всякие
data['Дата'] = data['Дата'].str.strip()
data['Расстояние'] = pd.to_numeric(data['Расстояние'], errors='coerce')
data['Расход бензина'] = pd.to_numeric(data['Расход бензина'], errors='coerce')
data['Масса груза'] = pd.to_numeric(data['Масса груза'], errors='coerce')

distance_7_to_9 = data[data['Дата'].isin(['7 октября', '8 октября', '9 октября'])]['Расстояние'].sum()

average_mass_from_osinki = data[data['Пункт отправления'] == 'Осинки']['Масса груза'].mean()

fuel_1_to_3 = data[data['Дата'].isin(['1 октября', '2 октября', '3 октября'])]['Расход бензина'].sum()

average_mass_to_berezki = data[data['Пункт назначения'] == 'Березки']['Масса груза'].mean()

print("Суммарное расстояние с 7 по 9 октября:", distance_7_to_9)
print("Средняя масса груза из Осинки:", average_mass_from_osinki)
print("Суммарный расход бензина с 1 по 3 октября:", fuel_1_to_3)
print("Средняя масса груза в Березки:", average_mass_to_berezki)
