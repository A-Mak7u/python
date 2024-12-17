import csv
import argparse


def read_file(file_name):
    """Чтение CSV-файла"""
    try:
        with open(file_name, encoding='utf-8') as file:
            reader = csv.reader(file)
            data = [row for row in reader]
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_name} не найден.")
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла {file_name}: {e}")


def parse_parameters():
    """Парсинг параметров из ввода"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='input.csv', help='Имя CSV-файла')
    parser.add_argument('--number', '-n', type=int, required=True, help='Число ящиков')
    parser.add_argument('--rows', type=int, required=True, help='Число непустых рядов')
    parser.add_argument('--cols', type=int, required=True, help='Число непустых колонок')
    parser.add_argument('--pigeons', '-m', type=int, required=True, help='Число предметов')
    parser.add_argument('--items', help='Предметы через запятую или точку с запятой')
    args = parser.parse_args()

    return args


def validate_data(data, rows, cols):
    """Проверка данных из CSV"""
    actual_rows = sum(1 for row in data if any(cell.strip() for cell in row))
    actual_cols = max((len(row) for row in data), default=0)

    errors = []
    if actual_rows != rows:
        errors.append(f"Указано {rows} непустых рядов, но найдено {actual_rows}.")
    if actual_cols != cols:
        errors.append(f"Указано {cols} непустых колонок, но найдено {actual_cols}.")

    return errors


def formulate_principle(boxes, items):
    """Формулировка принципа Дирихле"""
    if items > boxes:
        min_items = (items + boxes - 1) // boxes
        return f"Если в {boxes} ящиках лежит {items} предметов, то хотя бы в одном ящике лежит не менее {min_items} предметов."
    elif boxes > items:
        empty_boxes = boxes - items
        return f"Если в {boxes} ящиках лежит {items} предметов, то пустых ящиков как минимум {empty_boxes}."
    else:
        return f"Если в {boxes} ящиках лежит {items} предметов, то в каждом ящике лежит ровно один предмет."


def main():
    args = parse_parameters()

    try:
        data = read_file(args.file)

        errors = validate_data(data, args.rows, args.cols)
        if errors:
            print("Ошибки в данных:")
            for error in errors:
                print(f"- {error}")
            return

        principle = formulate_principle(args.number, args.pigeons)
        print(principle)

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()
