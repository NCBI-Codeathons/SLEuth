import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, draw, figure, cm
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
sns.set_style("whitegrid", {'axes.grid' : False})


def plot_elbow(min_num, max_num, distortions, y_label, title):
    """
    choose the number of clusters by the elblow plot
    """
    plt.plot(range(min_num, max_num), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()

def d3_tsne(X, y_km, heat=None):
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig) # Method 1
    # ax = fig.add_subplot(111, projection='3d') # Method 2
    X_embedded = TSNE(n_components=3, random_state=1234).fit_transform(X)
    #print(np.shape(X_embedded))
    x, y, z = X_embedded[:,0], X_embedded[:,1], X_embedded[:,2]
    ax.scatter(x, y, z, marker='o',c=y_km, cmap=plt.cm.get_cmap('Set1', 5))
    #plt.colorbar(ticks=range(1,6), label=y_km)
    ax.set_xlabel('tSNE-X')
    ax.set_ylabel('tSNE-Y')
    ax.set_zlabel('tSNE-Z')
    plt.show()

def d3_umap(X, y_km, heat=None):
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig) # Method 1
    # ax = fig.add_subplot(111, projection='3d') # Method 2
    reducer = umap.UMAP(random_state=1234, n_components=3)
    X_embedded = reducer.fit_transform(X)
    x, y, z = X_embedded[:,0], X_embedded[:,1], X_embedded[:,2]
    ax.scatter(x, y, z, marker='o',c=y_km, cmap=plt.cm.get_cmap('Set1', 5))
    #plt.colorbar(ticks=range(1,6), label=y_km)
    ax.set_xlabel('UMAP-X')
    ax.set_ylabel('UMAP-Y')
    ax.set_zlabel('UMAP-Z')
    plt.show()
    return reducer

def new_d3_umap(X, y_km, reducer,heat=None):
    fig = plt.figure(figsize=(6,6))
    ax = Axes3D(fig) # Method 1
    # ax = fig.add_subplot(111, projection='3d') # Method 2
    #reducer = umap.UMAP(random_state=1234, n_components=3)
    #X_embedded = reducer.fit_transform(X)
    X_embedded = reducer.transform(X)
    x, y, z = X_embedded[:,0], X_embedded[:,1], X_embedded[:,2]
    #1f77b4 -> blue
    #d62728 -> red
    #808080 -> grey
    ax.scatter(x, y, z, marker='o',c=['#808080']*(len(y_km)-4) +['#d62728']*2+['#1f77b4']*2, cmap=plt.cm.get_cmap('Set1', 5), s=[20]*(len(y_km)-4) + [200]*4)
    #2, 0, 1
    #ax.text(x[-4],y[-4],z[-4],  '   dsDNA', size=20, zorder=1,  color='k')
    #ax.text(x[-3],y[-3],z[-3],  '   dsDNA', size=20, zorder=1,  color='k')
    #ax.text(x[-2],y[-2],z[-2],  '   noDNA', size=20, zorder=1,  color='k')
    #ax.text(x[-1],y[-1],z[-1],  '   noDNA', size=20, zorder=1,  color='k')
    #plt.colorbar(ticks=range(1,6), label=y_km)
    ax.set_xlabel('UMAP-X')
    ax.set_ylabel('UMAP-Y')
    ax.set_zlabel('UMAP-Z')
    plt.show()


def vis_tsne(X,y_km):
    """
    visualize final clusters
    """
    X_embedded = TSNE(n_components=2, random_state=1234).fit_transform(X)
    #markers = ('o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    #random_heat = np.random.random_sample((np.shape(X)[0],))
    random_heat = np.random.normal(0, 0.2, np.shape(X)[0])
    #,
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full',  hue=y_km, palette="Set1")
    plt.title('Visualized by t-SNE')
    plt.savefig('figures/tSNE.pdf')
    plt.show()


def vis_umap(X, y_km):
    """
    visualize by UMAP
    """
    reducer = umap.UMAP(random_state=1234)
    X_embedded = reducer.fit_transform(X)
    #markers = ('o', 'v', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], legend='full',  hue=y_km, palette="Set1")
    plt.title('Visualized by UMAP')
    plt.savefig('figures/UMAP.pdf')
    plt.show()

def csv2X(fname):
    """
    input: csv files
    output: X array and each row is a patient
    """
    mypd = pd.read_csv(fname, sep=',')
    #row = list(mypd)
    #X = np.transpose(mypd.values)
    X = mypd.values
    #print(np.shape(X))
    X = [item[1:] for item in X]
    return X

def patient_cluster(patient_id, cluster_arr, num_clu, method):
    """
    input: csv file of patient id, numpy array of cluster membership
    output: csv file, first column: patient id; second column: cluster membership
    """
    with open(patient_id) as in_f:
        pid = list(in_f.readlines())
    cluster = np.load(cluster_arr)
    p_c = list(zip(pid, cluster))
    new_df = pd.DataFrame()
    new_df['patient_id'] = pid
    new_df['cluster'] = cluster
    new_df.to_csv("%s_patient_cluster%d.csv"%(method,num_clu),index=None)


if __name__ == "__main__":
    dat_type ="train"
    fname = "/Users/yuexichen/Desktop/School/UThackathon/datafiles/training_pca_projs.csv"
    X = csv2X(fname)
    #print(np.shape(X))
    num_clu = 5
    method = "kmeans"
    #method = "kmeans"
    np.save("%s_pca_good_projection"%dat_type,X)
    """
    with open("patients_id.txt",'w') as out_f:
        for r in row:
            out_f.write(r + '\n')
    patient_id = "patients_id.txt"
    #cluster_arr = "kmeans_membership_5clusters.npy"
    cluster_arr = "%s_membership_%d.npy"%(method,num_clu)
    patient_cluster(patient_id, cluster_arr, num_clu, method)
    """


