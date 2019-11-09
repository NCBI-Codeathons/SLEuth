import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris

def plot_elbow(min_num, max_num, distortions, y_label, title):
    """
    choose the number of clusters by the elblow plot
    """
    plt.plot(range(min_num, max_num), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def vis_tsne(X,y_km):
    """
    visualize final clusters
    """
    X_embedded = TSNE(n_components=2).fit_transform(X)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y_km, legend='full', palette="Set1")
    plt.show()
