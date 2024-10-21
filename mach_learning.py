def has_all_unique_digits(number):
    number_str = str(abs(number))
    return len(set(number_str)) == len(number_str)

N = int(input())
array = list(map(int, input().split()))

result = [num for num in array if not has_all_unique_digits(num)]

print(" ".join(map(str, result)))
