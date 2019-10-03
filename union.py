import csv
import sys
import string
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
                content=str(row['content']).lower()
                print(content)
                translator = str.maketrans('', '', string.punctuation)
                content.translate(translator)
                writer.writerow(
                    {'president': president,
                     'title': row['title'],
                     'date': row['date'],
                     'content': content,
                     'size': len(content.split()),
                     }
                )
