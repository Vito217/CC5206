import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

random_state = 0
with open('stopwords_es.txt','r') as f:
	stopwords_es = f.read().split('\n')

true_k = 6
data = pd.read_csv('smerge.csv')
vec = TfidfVectorizer(stop_words=stopwords_es)
vec.fit(data['sentence'].apply(lambda x: np.str_(x)))
features = vec.transform(data['sentence'].apply(lambda x: np.str_(x)))
#cls = MiniBatchKMeans(n_clusters=6, random_state=random_state)
print('initiatin kmeans')
k_means  = cluster.KMeans(n_clusters=true_k, max_iter=100, n_init=1)
print('kmeans initiated, training model...')
k_means.fit(features)
print('model traines, predicting')
##predict clusters
k_means.predict(features)
print('predicted, writing labels')
##cluster labels
#print(k_means.labels_)
print('so far so good')
##reduce feartures to 2D
#print('trying to graph')
#pca = PCA(n_components = 2,random_state=random_state)
print('so far so good ')
#reduced_features = pca.fit_transform(features.toarray())
print('so far so good')
##reduce the cluster centers to 2D
#reduced_cluster_centers = pca.transform(k_means.cluster_centers_)
print('so far so good, ready to graph')
#plt.scatter(reduced_features[:,0],reduced_features[:,1],c = k_means.predict(features))
#plt.scatter(reduced_cluster_centers[:,0],reduced_cluster_centers[:,1], marker='x',s=150,c='b')

print('writing the features of centroids')
termsFile = open('termsFile.txt','w+')
order_centroids= k_means.cluster_centers_.argsort()[:,::-1]
terms =vec.get_feature_names()
for i in terms:
	print(i, file=termsFile)
termsFile.close()

print('writing centroid in belonging cluster')
clusterCentroids = open('clusterCentroids.txt','w+')
for i in range(true_k):
	print('Cluster %d',i, file = clusterCentroids)
	for ind in order_centroids[i, :]:
		print(' %s' % terms[ind], file=clusterCentroids)
clusterCentroids.close()