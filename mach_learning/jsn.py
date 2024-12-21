import json
import csv

with open('constellations.json', 'r') as json_file:
    data = json.load(json_file)

stars = []

for constellation in data['constellations']:
    for star in constellation['brightest_stars']:
        star_info = {
            'star_name': star['name'],
            'brightness': star['brightness'],
            'constellation_name': constellation['latin_name'],
            'constellation_abbreviation': constellation['abbreviation'],
            'constellation_area': constellation['area'],
            'constellation_neighbors': ', '.join(constellation['neighboring_constellations'])
        }
        stars.append(star_info)

fields = ['star_name', 'brightness', 'constellation_name', 'constellation_abbreviation', 'constellation_area', 'constellation_neighbors']

with open('stars.csv', 'w', newline='') as out_file:
    writer = csv.DictWriter(out_file, fieldnames=fields)
    writer.writeheader()
    for star in stars:
        writer.writerow(star)
