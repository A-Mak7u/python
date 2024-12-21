import csv
import os
import chardet

def parse_input():
    params = {}
    items = []
    try:
        while True:
            line = input("Введите параметры (пустая строка для завершения): ").strip()
            if not line:
                break
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
            else:
                items.extend(
                    item.strip() for item in line.replace(";", ",").split(",")
                )
    except EOFError:
        pass

    params.setdefault("file", "input.csv")  # Путь к загруженному файлу
    params["items"] = items
    return params

def parse_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")

    # кодировк
    with open(file_path, 'rb') as f:
        encoding = chardet.detect(f.read())['encoding']

    with open(file_path, 'r', encoding=encoding) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        data = [row for row in reader if any(cell.strip() for cell in row)]

    if not data:
        raise ValueError(f"Файл {file_path} пустой.")

    return data

# проверка данных из CSV и параметров
def check_data(params, csv_data):
    errors = []
    try:
        n = int(params.get("n", 0))
        m = int(params.get("m", 0))
        rows = int(params.get("rows", 0))
        cols = int(params.get("cols", 0))
    except ValueError:
        raise ValueError("Параметры n, m, rows и cols должны быть целыми числами.")


    actual_rows = len(csv_data)
    actual_cols = max(len(row) for row in csv_data)

    if actual_rows != rows:
        errors.append(f"Количество строк в файле ({actual_rows}) не совпадает с rows={rows}.")
    if actual_cols != cols:
        errors.append(f"Количество столбцов в файле ({actual_cols}) не совпадает с cols={cols}.")
    if rows * cols != n:
        errors.append(f"Общее количество ящиков (rows * cols) ({rows * cols}) не совпадает с n={n}.")

    csv_items = [
        item.strip() for row in csv_data for cell in row if cell for item in cell.split(",")
    ]
    if len(csv_items) != m:
        errors.append(f"Количество предметов в файле ({len(csv_items)}) не совпадает с m={m}.")

    input_items = [item.strip() for item in params.get("items", [])]
    missing_items = set(input_items) - set(csv_items)
    if missing_items:
        errors.append(f"Предметы, отсутствующие в файле: {', '.join(missing_items)}.")

    return errors

def dirichle(n, m):
    if m > n:
        return f"Если в {n} ящиках лежит {m} предметов, то хотя бы в одном ящике лежит не менее {m // n + 1} предметов."
    elif m < n:
        return f"Если в {n} ящиках лежит {m} предметов, то пустых ящиков как минимум {n - m}."
    else:
        return "Все предметы могут быть распределены равномерно."

params = parse_input()
file_path = params["file"]

try:
    csv_data = parse_csv(file_path)
    errors = check_data(params, csv_data)

    if errors:
        print("Ошибки:")
        for error in errors:
            print("-", error)
    else:
        n = int(params["n"])
        m = int(params["m"])
        print(dirichle(n, m))
except Exception as e:
    print(f"Ошибка: {e}")
