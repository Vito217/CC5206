import csv
import sys
import string
presidents = ['pinera', 'bachelet', 'allende', 'macri', 'kirchner', 'fernandez']

with open('unionssplit.csv', 'w') as csvfile:
    fieldnames = ['president', 'sentence']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csv.field_size_limit(sys.maxsize)
    for president in presidents:
        with open(president+ '_ssplit.csv') as csvfile:
            readCSV = csv.DictReader(csvfile)
            for row in readCSV:
                print(president)
                sentence=str(row['sentence']).lower()
                translator = str.maketrans('', '', string.punctuation)
                sentence.translate(translator)
                writer.writerow(
                    {'president': president,
                     'sentence':sentence
                     }
                )
