import os
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import PCA as my_PCA
import random as rand
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift, SpectralClustering, \
    AgglomerativeClustering, DBSCAN, OPTICS, Birch
from nltk.corpus import stopwords
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from datetime import datetime

markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "+", "X", "d", "1", "2", "3", "4"]
colors = np.array(["red", "blue", "green", "yellow", "purple", "pink", "orange",
                   "black", "gray", "brown", "cyan", "magenta"])


def load_stopwords(string):
    """
    Lee archivos .txt con stopwords. Deben estar separadas por salto de linea
    :param string:
    :return:
    """
    print("Reading Stopwords")
    # Si el archivo existe
    if os.path.isfile(string):
        with open(string, 'r', encoding="utf8") as words:
            sw = words.read().split('\n')
    # Si el nombre es english o spanish
    else:
        sw = stopwords.words(string)
    return sw


def load_data(path, subpath, nrows=None, ignore=None, filt=None, oversampling=False, undersampling=False):
    """
    Carga datos desde un path, generalmente data/text, y un subpath, generalmente raw.
    Se puede ignorar presidentes y filtrar segun una expresion regular.
    Returna una tupla data, labels
    :param path:
    :param subpath:
    :param nrows:
    :param ignore:
    :param filt:
    :return:
    """
    print("Reading all data and labels")

    # Lista que tiene tuplas (discurso, persidente)
    data_labels = []
    for subdir, dirs, files in os.walk(path):
        for file in files:

            # Params es de la forma (dir, subdir, file)
            params = subdir.replace("/", "\\").split("\\")[1:]

            # No se lee el archivo si:
            # * Esta en ignore
            # * Si no esta en filter
            # * Si tiene alguna extension
            if not(file.endswith(".py") or file.endswith(".txt")) \
                    and (len(params) == 3 and params[2] == subpath):
                if not((ignore is not None and params[1] in ignore) or
                        (filt is not None and not filt.search(file))):
                    with open(os.path.join(subdir, file), 'r', encoding='utf8') as f:
                        # Se guarda la tupla (discurso, presidente)
                        text = f.read()
                        data_labels.append([text, params[1]])

    # Permutacion al azar
    shuffle(data_labels)

    # Si se especifico un numero de filas
    if nrows is not None:
        data_labels = data_labels[:nrows]

    # Separamos data y labels
    data = [corpus for corpus, _ in data_labels]
    labels = [label for _, label in data_labels]
    if oversampling or undersampling:
        if oversampling:
            sampler = RandomOverSampler()
        else:
            sampler = RandomUnderSampler()
        data, labels = sampler.fit_resample(np.reshape(data, (-1, 1)), np.reshape(labels, (-1, 1)))
        data, labels = np.reshape(data, (-1, )), np.reshape(labels, (-1, ))
    size = [len(text) for text in data]
    return data, labels, np.reshape(normalize(np.reshape(size, (-1, 1)), axis=0), (-1, ))*1000


def get_number_of_clusters(labels):
    """
    Retorna numero de clusters equivalente al numero de labels
    :param labels:
    :return:
    """
    print("Computing number of clusters")
    unique = np.unique(labels)
    return len(unique)


def vectorize(data, stop_words, ngram_min, ngram_max):
    """
    Calcula el TF-IDF de una matriz de textos. Incluye rango de ngramas
    :param data:
    :param stop_words:
    :param ngram_min:
    :param ngram_max:
    :return:
    """
    print("Getting Bag Of Words")
    vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(ngram_min, ngram_max))
    x = vectorizer.fit_transform(data)
    return x, vectorizer


def compute_centroids(x, lb):
    """
    Computa los centroides de cada cluster
    :param x:
    :param lb:
    :return:
    """

    clusters_dict = {}
    for c in np.unique(lb):
        clusters_dict[c] = []
    for i in range(len(x)):
        clusters_dict[lb[i]].append(x[i])
    centroids = None
    for cluster in clusters_dict.keys():
        centroid = np.mean(clusters_dict[cluster], axis=0)
        if centroids is None:
            centroids = np.array([centroid])
        else:
            centroids = np.append(centroids, [centroid], axis=0)

    return centroids


def get_eps_and_samples(x, k):
    """

    :param x:
    :param k:
    :return:
    """

    arr = np.empty((x.shape[0], 1))
    for i in range(x.shape[0]):
        substract = x - x[i]
        square = np.square(substract)
        sum = np.sum(square, axis=1)
        distances = np.sqrt(sum)
        distances = np.sort(distances)
        arr[i] = distances[k]
    arr = np.sort(arr, axis=0)
    plt.figure()
    plt.plot(arr)
    plt.show()


def clusters_frecuent_terms(x, vectorizer, its, n_clusters, cluster_type="kmeans", k_neigh=10):
    """
    Muestra los terminos mas frecuentes por cluster
    :param x:
    :param vectorizer:
    :param its:
    :param n_clusters:
    :param cluster_type:
    :return:
    """

    print("Getting most frecuent terms per cluster")

    # Elegimos el tipo de clustering
    if cluster_type == "mbkmeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, n_init=its)
        title = "MINI BATCH KMEANS----------------------------------------------------------"
    elif cluster_type == "affprop":
        km = AffinityPropagation()
        title = "AFFINITY PROPAGATION-------------------------------------------------------"
    elif cluster_type == "mshift":
        km = MeanShift()
        title = "MEAN SHIFT-----------------------------------------------------------------"
    elif cluster_type == "spec":
        km = SpectralClustering(n_clusters=n_clusters, n_init=its)
        title = "SPECTRAL-------------------------------------------------------------------"
    elif cluster_type == "aggc":
        km = AgglomerativeClustering(n_clusters=n_clusters)
        title = "AGGLOMERATIVE--------------------------------------------------------------"
    elif cluster_type == "dbscan":
        km = DBSCAN(eps=0.1, min_samples=k_neigh)
        title = "DBSCAN---------------------------------------------------------------------"
    elif cluster_type == "optisc":
        km = OPTICS()
        title = "OPTICS---------------------------------------------------------------------"
    elif cluster_type == "birch":
        km = Birch()
        title = "BIRCH----------------------------------------------------------------------"
    else:
        km = KMeans(n_clusters=n_clusters, n_init=its)
        title = "KMEANS---------------------------------------------------------------------"

    print(title)
    km.fit(x.toarray())
    lb = km.labels_
    true_k = np.unique(lb).shape[0]
    print(true_k)
    if cluster_type not in ["spec", "aggc", "dbscan", "optisc", "birch"]:
        order_centroids = km.cluster_centers_
        if cluster_type not in ["kmeans", "mbkmeans", "spec", "aggc", "mshift"]:
            order_centroids = order_centroids.toarray()
    else:
        order_centroids = compute_centroids(x.toarray(), lb)
    order_centroids = order_centroids.argsort()[:, ::-1]
    global colors
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        if i < colors.shape[0]:
            color = colors[i]
        else:
            color = '#%02X%02X%02X' % (rand.randint(0, 255), rand.randint(0, 255), rand.randint(0, 255))
            colors = np.append(colors, np.array([color]))
        print("Cluster {}:".format(color), end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()


def plot_clusters(x, labels, size, dim, its, n_clusters, cluster_type="kmeans", save=False, trunc_method="SPCA",
                  k_neigh=10):
    """
    Grafica los clusters.
    :param x:
    :param labels:
    :param dim:
    :param n_clusters:
    :param cluster_type:
    :param save:
    :return:
    """

    print("Plotting Clusters")

    # Se calculan los clusters. Los centroides se almacenan en km
    if trunc_method == "SPCA" or trunc_method == "TSVD":
        if trunc_method == "SPCA":
            trunc = PCA(n_components=dim)
            x = x.toarray()
        else:
            trunc = TruncatedSVD(n_components=dim)
        x = trunc.fit_transform(x)
    else:
        x = my_PCA.pca(x, dim)

    # Elegimos el tipo de clustering
    if cluster_type == "mbkmeans":
        km = MiniBatchKMeans(n_clusters=n_clusters, n_init=its)
    elif cluster_type == "affprop":
        km = AffinityPropagation()
    elif cluster_type == "mshift":
        km = MeanShift()
    elif cluster_type == "spec":
        km = SpectralClustering(n_clusters=n_clusters, n_init=its)
    elif cluster_type == "aggc":
        km = AgglomerativeClustering(n_clusters=n_clusters)
    elif cluster_type == "dbscan":
        get_eps_and_samples(x, k_neigh)
        km = DBSCAN(eps=0.1, min_samples=k_neigh)
    elif cluster_type == "optisc":
        km = OPTICS()
    elif cluster_type == "birch":
        km = Birch(n_clusters=n_clusters, threshold=0.1)
    else:
        km = KMeans(n_clusters=n_clusters, n_init=its)

    # LB son los indices del cluster al que pertenece cada fila de datos
    km.fit(x)
    lb = km.labels_
    dendograma = False
    if cluster_type == "aggc":
        if dendograma:
            data = km.children_
            Z = linkage(data)
            dendrogram(Z,p=n_clusters,labels=lb, truncate_mode='lastp',get_leaves=True, count_sort='ascending')  


    # Como se hace mas de un plot a la vez, separamos la data por presidente
    data_dict = {}
    marker_ind = 0
    for i in range(len(x)):

        # Se le asigna un marcador y se guarda una lista de sus discursos y su respectivo cluster
        label = labels[i]
        if label not in data_dict:
            data_dict[label] = {'data': [], 'clusters': [], 'sizes': [], 'marker': markers[marker_ind]}
            marker_ind += 1
        data_dict[label]['data'].append(x[i])
        data_dict[label]['clusters'].append(lb[i])
        data_dict[label]['sizes'].append(size[i])

    # Si el grafico es en 2D
    if dim == 2:

        plt.figure()
        # Iteramos y hacemos un plot por cada presidente
        for key in data_dict.keys():
            x = [row[0] for row in data_dict[key]['data']]
            y = [row[1] for row in data_dict[key]['data']]
            c = data_dict[key]['clusters']
            s = data_dict[key]['sizes']
            marker = data_dict[key]['marker']
            l = plt.scatter(x, y, c=colors[c], s=s, marker=marker, label=key, edgecolors="black")
        plt.legend()
        plt.show()

        # Guardamos el plot
        if save:
            if not os.path.exists("results/speech_clustering"):
                os.makedirs("results/speech_clustering")
            plt.savefig("results/speech_clustering/{}.png".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

    # Si el grafico es en 3D
    elif dim == 3:

        plot = plt.figure()
        ax = Axes3D(plot)

        # Iteramos y hacemos un plot por cada presidente
        for key in data_dict.keys():
            x = [row[0] for row in data_dict[key]['data']]
            y = [row[1] for row in data_dict[key]['data']]
            z = [row[2] for row in data_dict[key]['data']]
            c = data_dict[key]['clusters']
            s = data_dict[key]['sizes']
            marker = data_dict[key]['marker']
            ax.scatter(x, y, z, c=colors[c], s=s, marker=marker, label=key, edgecolors="black")
        plot.legend()
        plot.show()

        # Guardamos el plot
        if save:
            if not os.path.exists("results/speech_clustering"):
                os.makedirs("results/speech_clustering")
            plot.savefig("results/speech_clustering/{}.png".format(datetime.now().strftime("%Y%m%d-%H%M%S")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate clusters")
    parser.add_argument("--oversampling", action='store_true')
    parser.add_argument("--undersampling", action='store_true')
    parser.add_argument("-stopwords", type=str, help="Path to stopwords, or english or spanish", required=True)
    parser.add_argument("-data", type=str, help="Path to dataset. Every folder name is the label.", required=True)
    parser.add_argument("-subdata", type=str, help="Subdata available: raw, anio, anio_mes, merge", required=True)
    parser.add_argument("-ngram_min", type=int, help="Min range of n-gram", required=True)
    parser.add_argument("-ngram_max", type=int, help="Max range of n-gram", required=True)
    parser.add_argument("-dim", type=int, help="2 for 2D plot, 3 for 3D plot", required=True)
    parser.add_argument("-type", type=str, help="List of methods (e.g. kmeans,aggc,dbscan)",
                        required=False)
    parser.add_argument("-nrows", type=int, help="Number of rows to use", required=False)
    parser.add_argument("-ignore", type=str, help="CSV list with candidates (e.g. pinera,bachelet)", required=False)
    parser.add_argument("-filter", type=str, help="Regex used to filter some texts", required=False)
    parser.add_argument("-nclusters", type=int, help="Number of clusters to use", required=False)
    parser.add_argument("-iterations", type=int, help="Number of iterations", required=False)
    parser.add_argument("-trunc_method", type=str, help="PCA, SPCA (Scikit-learn PCA) or TSVD", required=False)
    parser.add_argument("--save", action='store_true')

    pargs = parser.parse_args()
    stopwords = load_stopwords(pargs.stopwords)
    ignore = pargs.ignore.split(",") if pargs.ignore else None
    nrows = pargs.nrows if pargs.nrows else None
    filt = re.compile(pargs.filter) if pargs.filter else None
    over = True if pargs.oversampling else False
    under = True if pargs.undersampling else False
    its = pargs.iterations if pargs.iterations else 10
    data, labels, size = load_data(pargs.data, pargs.subdata, nrows, ignore, filt,
                                   oversampling=over, undersampling=under)
    n_clusters = pargs.nclusters if (pargs.nclusters and pargs.nclusters >= 1) else get_number_of_clusters(labels)
    cluster_type = pargs.type.split(",") if pargs.type else ["kmeans"]
    trunc_method = pargs.trunc_method if pargs.trunc_method else "SPCA"
    ngram_min = pargs.ngram_min
    ngram_max = pargs.ngram_max
    x, vectorizer = vectorize(data, stopwords, ngram_min, ngram_max)
    dim = pargs.dim
    save = pargs.save
    for method in cluster_type:
        clusters_frecuent_terms(x, vectorizer, its, n_clusters, method)
        plot_clusters(x, labels, size, dim, its, n_clusters, method, save, trunc_method)
