import csv
import sys

presidents = ['pinera', 'bachelet', 'allende', 'macri', 'kirchner', 'fernandez']

with open('data/csv/union.csv', 'w') as csvfile:
    fieldnames = ['president', 'size','count', 'title', 'date', 'content']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csv.field_size_limit(sys.maxsize)

    for president in presidents:
        with open("data/csv/"+president + '.csv') as csvfile:
            readCSV = csv.DictReader(csvfile)
            for row in readCSV:
                print(row)

                writer.writerow(
                    {'president': president,
                     'title': row['title'],
                     'date': row['date'],
                     'content': row['content'],
                     'size': len(row['content']),
                     }
                )
