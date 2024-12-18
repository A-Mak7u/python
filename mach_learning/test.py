from itertools import accumulate

# Читаем входное значение
n = int(input())

# Вычисляем суммы первых i чисел с помощью accumulate
sums = accumulate(range(1, n + 1))

# Вычисляем итоговую сумму по описанной формуле
total_sum = sum((-1) ** (i + 1) * s ** 2 for i, s in enumerate(sums))

# Выводим результат
print(total_sum)
