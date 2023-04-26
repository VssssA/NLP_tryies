from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# загрузка корпуса текстов и обучение модели Word2Vec
sentences = [['this', 'is', 'a', 'sentence'], ['another', 'sentence']]
model = Word2Vec(sentences, min_count=1)

# получение эмбеддингов слов
word_vectors = model.wv

# кластеризация эмбеддингов с помощью K-means
num_clusters = 2
kmeans_model = KMeans(n_clusters=num_clusters)
kmeans_model.fit(word_vectors.vectors)

# получение центров кластеров
cluster_centers = kmeans_model.cluster_centers_

# вычисление попарных расстояний между центрами кластеров

distance_matrix = pdist(cluster_centers)

# сравнение удаленности центров кластеров
cosine_distance_matrix = cosine_distances(cluster_centers)
euclidean_distance_matrix = euclidean_distances(cluster_centers)

print("Расстояние между центрами кластеров:")
print(distance_matrix)
print("Косинусное расстояние:")
print(cosine_distance_matrix)
print("Евклидово расстояние:")
print(euclidean_distance_matrix)
