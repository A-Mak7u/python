def mod(lst):
    for i in range(len(lst)):
        if lst[i] % 2 == 0:
            lst[i] *= 2


n = [1, 2, 3, 4, 5, 6]
mod(n)
print(n)
