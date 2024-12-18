import csv

file_path = "task19.csv"

distance_7_to_9 = 0
fuel_1_to_3 = 0
mass_from_osinki = []
mass_to_berezki = []

with open(file_path, mode='r', encoding='windows-1251') as file:
    reader = csv.DictReader(file, delimiter=';')
    for row in reader:
        date = row['Дата'].strip()
        try:
            distance = float(row['Расстояние'])
        except ValueError:
            distance = 0
        try:
            fuel = float(row['Расход бензина'])
        except ValueError:
            fuel = 0
        try:
            cargo_mass = float(row['Масса груза'])
        except ValueError:
            cargo_mass = None

        if date in ['7 октября', '8 октября', '9 октября']:
            distance_7_to_9 += distance

        if date in ['1 октября', '2 октября', '3 октября']:
            fuel_1_to_3 += fuel

        if row['Пункт отправления'] == 'Осинки' and cargo_mass is not None:
            mass_from_osinki.append(cargo_mass)

        if row['Пункт назначения'] == 'Березки' and cargo_mass is not None:
            mass_to_berezki.append(cargo_mass)

average_mass_from_osinki = sum(mass_from_osinki) / len(mass_from_osinki) if mass_from_osinki else 0
average_mass_to_berezki = sum(mass_to_berezki) / len(mass_to_berezki) if mass_to_berezki else 0

print("Суммарное расстояние с 7 по 9 октября:", distance_7_to_9)
print("Средняя масса груза из Осинки:", average_mass_from_osinki)
print("Суммарный расход бензина с 1 по 3 октября:", fuel_1_to_3)
print("Средняя масса груза в Березки:", average_mass_to_berezki)
