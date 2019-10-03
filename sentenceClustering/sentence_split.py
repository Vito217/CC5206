import csv
import sys
import spacy
import os
csv.field_size_limit(sys.maxsize)
nlp = spacy.load('es_core_news_sm')


allende = 'allende.csv'
pinera = 'pinera.csv'
bachelet = 'bachelet.csv'
fernandez = 'fernandez.csv'
kirchner = 'kirchner.csv'
macri = 'macri.csv'


def split_speeches():
    if not os.path.exists('allende_split.csv'):
        with open (allende ) as readFile:
            allende_split = open('allende_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    allende_split.write(line)
                else:
                    allende_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1


    if not os.path.exists('pinera_split.csv'):
        with open (pinera ) as readFile:
            pinera_split = open('pinera_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    pinera_split.write(line)
                else:
                    pinera_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1

    if not os.path.exists('bachelet_split.csv'):
        with open (bachelet) as readFile:
            bachelet_split = open('bachelet_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    bachelet_split.write(line)
                else:
                    bachelet_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1

    if not os.path.exists('fernandez_split.csv'):
        with open (fernandez ) as readFile:
            fernandez_split = open('fernandez_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    fernandez_split.write(line)
                else:
                    fernandez_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1

    if not os.path.exists('kirchner_split.csv'):
        with open (kirchner ) as readFile:
            kirchner_split = open('kirchner_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    kirchner_split.write(line)
                else:
                    kirchner_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1

    if not os.path.exists('macri_split.csv'):
        with open (macri ) as readFile:
            macri_split = open('macri_split.csv','a+')
            reader = csv.reader(readFile)
            count= 1
            for row in reader:
                if count == 2:
                    discurso = row[2]
                    sentences = nlp(discurso)
                    split_discurso =''
                    for sent in sentences.sents:
                        split_discurso = split_discurso + ',' + sent.text
                    line= row[0] + ',' + row[1] + split_discurso + '\n'
                    macri_split.write(line)
                else:
                    macri_split.write(row[0]+','+row[1]+','+row[2]+'\n')
                    count = count +1

def ssplit_speeches(president):
    split_file = open(president + '_ssplit.csv','w+')
    split_file.write('')
    split_file.close()
    with open (president + '.csv' ) as readFile:
        split_file = open(president + '_ssplit.csv','a+')
        reader = csv.reader(readFile)
        count= 1
        for row in reader:
            if count == 2:
                discurso = row[2]
                sentences = nlp(discurso)
                for sent in sentences.sents:
                    split_file.write(president+',' + sent.text + '\n')
            else:
                split_file.write('president,sentence\n')
                count = count +1
        split_file.close()
    readFile.close()
def ssplit():
    print('splitting pinera')
    ssplit_speeches('pinera')
    print('splitting bachelet')
    ssplit_speeches('bachelet')
    print('splitting allende')
    ssplit_speeches('allende')
    print('splitting macri')
    ssplit_speeches('macri')
    print('splitting fernandez')
    ssplit_speeches('fernandez')
    print('splitting kirchner')
    ssplit_speeches('kirchner')

def smerge():
    smerge = open('smerge.csv','w')
    smerge.write('')
    smerge.close()
    smerge = open('smerge.csv','a+')
    print('copiando pinera')
    pinerafile = open ('pinera_ssplit.csv') 
    pinerareader = csv.reader(pinerafile)
    for row in pinerareader:
        if len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
    pinerafile.close()
    print('copiando bachelet')

    bacheletfile = open('bachelet_ssplit.csv') 
    bacheletreader = csv.reader(bacheletfile)
    count = 1
    for row in bacheletreader:
        if count == 2 and len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
        else:
            count = count +1
    bacheletfile.close()
    print('copiando allende')

    allendefile = open('allende_ssplit.csv') 
    allendereader = csv.reader(allendefile)
    count = 1
    for row in allendereader:
        if count == 2 and len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
        else:
            count = count +1
    allendefile.close()
    print('copiando macri')

    macrifile = open('macri_ssplit.csv') 
    macrireader = csv.reader(macrifile)
    count = 1
    for row in macrireader:
        if count == 2 and len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
        else:
            count = count +1
    macrifile.close()
    print('copiando fernandez')

    fernandezfile = open('fernandez_ssplit.csv')
    ferreader = csv.reader(fernandezfile)
    count = 1
    for row in ferreader:
        if count == 2 and len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
        else:
            count = count +1
    fernandezfile.close()
    print('copiando kirchner')

    kirchnerfile = open('kirchner_ssplit.csv')
    kreader = csv.reader(kirchnerfile)
    count = 1
    for row in kreader:
        if count == 2 and len(row)>1:
            smerge.write(row[0] + ',' + row[1] + '\n')
        else:
            count = count +1
    kirchnerfile.close()
    smerge.close()

ssplit()
smerge() 
