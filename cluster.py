import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids

markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "+", "X", "d", "1", "2", "3", "4"]


def load_stopwords(string):
    print("Reading Stopwords")
    if os.path.isfile(string):
        with open(string, 'r', encoding="utf8") as words:
            sw = words.read().split('\n')
    else:
        sw = stopwords.words(string)
    return sw


def load_data(path, subpath, nrows):
    print("Reading all data and labels")
    data_labels = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if not(file.endswith(".py") or file.endswith(".txt")) \
                    and (len(subdir.split("\\")) == 3 and subdir.split("\\")[2] == subpath):
                with open(os.path.join(subdir, file), 'r', encoding='utf8') as f:
                    data_labels.append([f.read(), subdir.split("\\")[1]])
    shuffle(data_labels)
    data_labels = data_labels[:nrows]
    data = [corpus for corpus, _ in data_labels]
    labels = [label for _, label in data_labels]
    return data, labels


def get_number_of_clusters(labels):
    print("Indexing labels")
    le = LabelEncoder()
    target_ind = le.fit_transform(labels)
    n_clusters = np.unique(target_ind).shape[0]
    return n_clusters


def vectorize(data, stop_words, ngram_min, ngram_max):
    print("Getting Bag Of Words")
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(ngram_min, ngram_max))
    x = vectorizer.fit_transform(data)
    x = normalize(x)
    return x


def plot_clusters(x, labels, dim, n_clusters):
    trunc = TruncatedSVD(n_components=dim)
    x = trunc.fit_transform(x)
    km = KMeans(n_clusters=n_clusters)
    km.fit(x)
    lb = km.labels_
    data_dict = {}
    for i in range(len(x)):
        label = labels[i]
        if label not in data_dict:
            data_dict[label] = {'data': [], 'clusters': [], 'marker': markers.pop(0)}
        data_dict[label]['data'].append(x[i])
        data_dict[label]['clusters'].append(lb[i])
    legends = []
    if dim == 2:
        for key in data_dict.keys():
            x = [row[0] for row in data_dict[key]['data']]
            y = [row[1] for row in data_dict[key]['data']]
            c = [float(num) for num in data_dict[key]['clusters']]
            marker = data_dict[key]['marker']
            l = plt.scatter(x, y, c=c, marker=marker, label=key)
            legends.append(l)
        plt.legend(handles=legends)
        plt.show()
    elif dim == 3:
        plot = plt.figure()
        ax = Axes3D(plot)
        for key in data_dict.keys():
            x = [row[0] for row in data_dict[key]['data']]
            y = [row[1] for row in data_dict[key]['data']]
            z = [row[2] for row in data_dict[key]['data']]
            c = [float(num) for num in data_dict[key]['clusters']]
            marker = data_dict[key]['marker']
            l = ax.scatter(x, y, z, c=c, marker=marker, label=key)
            legends.append(l)
        plot.legend(handles=legends)
        plot.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate clusters")
    # parser.add_argument("--encode", action='store_true')
    parser.add_argument("--oversampling", action='store_true')
    parser.add_argument("--undersampling", action='store_true')
    parser.add_argument("-stopwords", type=str, help="Path to stopwords", required=True)
    parser.add_argument("-data", type=str, help="Path to dataset. Every folder name is the label.", required=True)
    parser.add_argument("-subdata", type=str, help="Subdata available: raw, anio, anio_mes", required=True)
    parser.add_argument("-ngram_min", type=int, help="Min range of n-gram", required=True)
    parser.add_argument("-ngram_max", type=int, help="Max range of n-gram", required=True)
    parser.add_argument("-dim", type=int, help="2 for 2D plot, 3 for 3D plot", required=True)
    parser.add_argument("-nrows", type=int, help="Number of rows to use", required=True)

    pargs = parser.parse_args()
    stopwords = load_stopwords(pargs.stopwords)
    data, labels = load_data(pargs.data, pargs.subdata, pargs.nrows)
    if pargs.oversampling:
        sampler = SMOTE()
        data, labels = sampler.fit_resample(data, labels)
    elif pargs.undersampling:
        sampler = ClusterCentroids()
        data, labels = sampler.fit_resample(data, labels)
    n_clusters = get_number_of_clusters(labels)
    ngram_min = pargs.ngram_min
    ngram_max = pargs.ngram_max
    x = vectorize(data, stopwords, ngram_min, ngram_max)
    dim = pargs.dim
    plot_clusters(x, labels, dim, n_clusters)
