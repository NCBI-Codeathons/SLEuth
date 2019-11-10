import sklearn
from sklearn.cluster import KMeans

from src.features.feature_selection import PCA_Variants2Gene_FeatureSelection


class SLEuth(sklearn.base.TransformerMixin, sklearn.base.ClusterMixin):
    def __init__(self, cluster_num, variants_genes_path="../../data/interim/variants_top56_genes.csv",
                 variance_threshold=0.9, init='random', n_init=10, max_iter=300,tol=1e-4,random_state=40):
        """
        Run fit_transform() to compute top_k PCA components first. Run transform() if need to get projections from SNP data.

        Then, run fit_predict() to run KMeans clustering on the PCA transformed data. To predict a SNP profile which
        cluster they belong to, run predict()

        :param cluster_num:
            The number of clusters to form as well as the number of centroids to generate
        :param variants_genes_path:
            Path to the csv file that provides two column, ["Gene(s)", "Variant ID"] that the user would like to
        :param variance_threshold:
            e.g. 0.9 for 90% variance. The variance threshold to select the minimum number of PCA components such that a % of the variance remains.
        :param init:
            'random': choose k observations (rows) at random from data for the initial centroids.
        :param n_init:
            Number of time the k-means algorithm will be run with different centroid seeds
        :param max_iter:
        :param tol:
        :param random_state:
        """
        self.pca_variants_fs = PCA_Variants2Gene_FeatureSelection(variants_genes_path, variance_threshold)
        self.km = KMeans(
            n_clusters=cluster_num, init=init,
            n_init=n_init, max_iter=max_iter,
            tol=tol, random_state=random_state
        )

    def fit_transform(self, X, y=None, **fit_params):
        """
        :param X: A Pandas DataFrame where row indices are patient samples and columns are SNP sites with
        :return:
        """
        return self.pca_variants_fs.fit_transform(X, y, **fit_params)

    def transform(self, X, y=None):
        return self.pca_variants_fs.transform(X, y)

    def fit_predict(self, X, y=None):
        X_transformed = self.transform(X)
        return self.km.fit_predict(X_transformed, y)

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.km.predict(X_transformed)