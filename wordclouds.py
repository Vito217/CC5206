import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud


def random_color_func(word=None, font_size=None, position=None, orientation=None, font_path=None, random_state=None):
    h = int(360.0 * 45.0 / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(60, 120)) / 255.0)

    return "hsl({}, {}%, {}%)".format(h, s, l)


def largo_texto(presidente):
    text_anio = ""
    contador = []
    with open('data/csv/union.csv', encoding="utf-8") as csvfile:
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        readCSV = csv.DictReader(csvfile)
        for row in readCSV:
            if row['president'] == presidente:
                contador.append(int(row['size']))
    return contador


print(np.mean(largo_texto('pinera')))
print(np.mean(largo_texto('bachelet')))

print(np.median(largo_texto('pinera')))
print(np.median(largo_texto('bachelet')))


def read_for_date(dir, fecha, par):
    text_anio = ""
    contador = 0
    with open(dir, encoding="utf-8") as csvfile:
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        readCSV = csv.DictReader(csvfile)
        for row in readCSV:
            if row['date'][:par] == fecha:
                contador += 1
                text_anio = text_anio + row['content']
    print(contador)
    return text_anio


def read_for_year(dir, anio):
    text_anio = ""
    contador = 0
    with open(dir, encoding="utf-8") as csvfile:
        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.

            try:
                csv.field_size_limit(maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        readCSV = csv.DictReader(csvfile)
        for row in readCSV:
            if row['date'][:4] == anio:
                contador = contador + 1
                text_anio = text_anio + row['content']
    print(contador)
    return text_anio


def generar_wordcloud2(filecontent, imgname):
    with open('stopwords_es.txt', 'r', encoding="utf8") as f:
        stopwords_es = f.read().split('\n')
    dir_name="results/wordcloud/"
    wordcloud = WordCloud(
                        stopwords=stopwords_es,
                        background_color='white',
                        width=1200,
                        height=1000,
                        color_func=random_color_func
                        ).generate(filecontent)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(dir_name+imgname)


# bachelet

generar_wordcloud2(read_for_year('data/csv/bachelet.csv', '2014'), "bachelet_2014")
generar_wordcloud2(read_for_year('data/csv/bachelet.csv', '2015'), "bachelet_2015")
generar_wordcloud2(read_for_year('data/csv/bachelet.csv', '2016'), "bachelet_2016")
generar_wordcloud2(read_for_year('data/csv/bachelet.csv', '2017'), "bachelet_2017")

# test
generar_wordcloud2(read_for_date('data/csv/bachelet.csv', '2017_05', 7), "bachelet_2017_05")
# pinera
generar_wordcloud2(read_for_year('data/csv/pinera.csv', '2010'), "pinera_2010")
generar_wordcloud2(read_for_year('data/csv/pinera.csv', '2011'), "pinera_2011")
generar_wordcloud2(read_for_year('data/csv/pinera.csv', '2012'), "pinera_2012")
generar_wordcloud2(read_for_year('data/csv/pinera.csv', '2013'), "pinera_2013")

generar_wordcloud2(read_for_date('data/csv/pinera.csv', '2010_03', 7), "pinera_2010_03")
generar_wordcloud2(read_for_date('data/csv/pinera.csv', '2010_04', 7), "pinera_2010_04")
generar_wordcloud2(read_for_date('data/csv/pinera.csv', '2010_05', 7), "pinera_2010_05")


def generar_wordcloud(filename, imgname):
    file_content = open(filename, encoding='utf-8').read()

    with open('stopwords_es.txt', 'r', encoding="utf8") as f:
        stopwords_es = f.read().split('\n')

    wordcloud = WordCloud(font_path=r'C:\Windows\Fonts\Verdana.ttf',
                          stopwords=stopwords_es,
                          background_color='white',
                          width=1200,
                          height=1000,
                          color_func=random_color_func
                          ).generate(file_content)

    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig(imgname)

# https://stackoverflow.com/questions/42418085/python-wordcloud-from-a-txt-file referencia de codigo
