a = input()

unique = set()

rep = set()

for char in a:
    if char in unique:
        rep.add(char)
        unique.discard(char)
    elif char not in rep:
        unique.add(char)

print(''.join(sorted(unique)))
