import random

def generate_id():
    while True:
        id_number = random.randint(10000, 99999)
        digits = [int(d) for d in str(id_number)]
        if sum(digits[1:4]) == sum([digits[0], digits[4]]):
            return id_number

def generate_login():
    glas = 'aeiou'
    soglas = ''.join(set('abcdefghijklmnopqrstuvwxyz') - set(glas))
    while True:
        login = ''.join(random.choices(glas + soglas, k=6))
        if all(login.count(char) <= 2 for char in login):
            return login

def generate_password():
    lt = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    while True:
        password = ''.join(random.sample(lt, k=10))
        if len(set(password.lower())) == 10:
            return password

def generate_unique_records(n):
    records = set()
    while len(records) < n:
        id_number = generate_id()
        login = generate_login()
        password = generate_password()
        if (id_number, login, password) not in records:
            records.add((id_number, login, password))
    return list(records)


records = generate_unique_records(5)
for record in records:
    print(record)
