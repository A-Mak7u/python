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
        print("Ошибка: неожиданное завершение ввода.")
    params.setdefault("file", "input.csv")
    params["items"] = items
    return params

def validate_params(params):
    try:
        params["n"] = int(params.get("n", 0))
    except ValueError:
        print("Ошибка: параметр 'n' должен быть целым числом.")
        raise
    try:
        params["m"] = int(params.get("m", 0))
    except ValueError:
        print("Ошибка: параметр 'm' должен быть целым числом.")
        raise
    try:
        params["rows"] = int(params.get("rows", 0))
    except ValueError:
        print("Ошибка: параметр 'rows' должен быть целым числом.")
        raise
    try:
        params["cols"] = int(params.get("cols", 0))
    except ValueError:
        print("Ошибка: параметр 'cols' должен быть целым числом.")
        raise

def parse_csv(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден.")
    except FileNotFoundError as e:
        print(e)
        raise

    try:
        with open(file_path, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
    except Exception as e:
        print(f"Ошибка определения кодировки: {e}")
        raise

    try:
        with open(file_path, 'r', encoding=encoding) as csv_file:
            reader = csv.reader(csv_file, delimiter=',', quotechar='"')
            data = [row for row in reader if any(cell.strip() for cell in row)]
        if not data:
            raise ValueError(f"Файл {file_path} пустой.")
    except ValueError as e:
        print(e)
        raise
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        raise

    return data

def check_data(params, csv_data):
    errors = []
    n, m, rows, cols = params["n"], params["m"], params["rows"], params["cols"]

    try:
        actual_rows = len(csv_data)
        if actual_rows != rows:
            errors.append(f"Количество строк в файле ({actual_rows}) не совпадает с rows={rows}.")
    except Exception as e:
        print(f"Ошибка при проверке строк: {e}")

    try:
        actual_cols = max(len(row) for row in csv_data)
        if actual_cols != cols:
            errors.append(f"Количество столбцов в файле ({actual_cols}) не совпадает с cols={cols}.")
    except Exception as e:
        print(f"Ошибка при проверке столбцов: {e}")

    try:
        if rows * cols != n:
            errors.append(f"Общее количество ящиков (rows * cols) ({rows * cols}) не совпадает с n={n}.")
    except Exception as e:
        print(f"Ошибка при проверке ящиков: {e}")

    try:
        csv_items = [
            item.strip() for row in csv_data for cell in row if cell for item in cell.split(",")
        ]
        if len(csv_items) != m:
            errors.append(f"Количество предметов в файле ({len(csv_items)}) не совпадает с m={m}.")
    except Exception as e:
        print(f"Ошибка при подсчете предметов: {e}")

    try:
        input_items = [item.strip() for item in params.get("items", [])]
        missing_items = set(input_items) - set(csv_items)
        if missing_items:
            errors.append(f"Предметы, отсутствующие в файле: {', '.join(missing_items)}.")
    except Exception as e:
        print(f"Ошибка при проверке предметов: {e}")

    return errors

def dirichle(n, m):
    try:
        if m > n:
            return f"Если в {n} ящиках лежит {m} предметов, то хотя бы в одном ящике лежит не менее {m // n + 1} предметов."
        elif m < n:
            return f"Если в {n} ящиках лежит {m} предметов, то пустых ящиков как минимум {n - m}."
        else:
            return "Все предметы могут быть распределены равномерно."
    except Exception as e:
        print(f"Ошибка в расчетах: {e}")
        raise

def main():
    try:
        params = parse_input()
        validate_params(params)
        file_path = params["file"]
        csv_data = parse_csv(file_path)
        errors = check_data(params, csv_data)

        if errors:
            print("Ошибки:")
            for error in errors:
                print("-", error)
        else:
            n = params["n"]
            m = params["m"]
            print(dirichle(n, m))
    except Exception as e:
        print(f"Общая ошибка: {e}")

if __name__ == "__main__":
    main()
