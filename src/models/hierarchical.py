import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from kmeans import vis_tsne

#'ward' or 'single'

def plot_dendro(X, link_method):
    """
    link_method: single or ward
    """
    labelList = range(len(np.shape(X)[0]))
    linked = linkage(X, link_method)
    dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
    plt.show()

def dendro_cluster(optimal_num, link_method):
    cluster = AgglomerativeClustering(n_clusters=optimal_num, affinity='euclidean', linkage=link_method)
    cluster.fit_predict(data)
    vis_tsne(X,cluster.labels_)

if __name__ == "__main__":
    link_method = 'ward'
    plot_dendro(X, link_method)
    optimal_num = 
    dendro_cluster(optimal_num, link_method)




