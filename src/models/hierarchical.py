import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from kmeans import vis_tsne
from sklearn.datasets import load_iris

#'ward' or 'single'

def plot_dendro(X, link_method):
    """
    link_method: single or ward
    """
    labelList = range(np.shape(X)[0])
    linked = linkage(X, link_method)
    dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()

def dendro_cluster(X, optimal_num, link_method):
    cluster = AgglomerativeClustering(n_clusters=optimal_num, affinity='euclidean', linkage=link_method)
    cluster.fit_predict(X)
    vis_tsne(X,cluster.labels_)

if __name__ == "__main__":
    X = load_iris().data
    link_method = 'ward'
    #plot_dendro(X, link_method)
    optimal_num = 6
    dendro_cluster(X, optimal_num, link_method)
    



