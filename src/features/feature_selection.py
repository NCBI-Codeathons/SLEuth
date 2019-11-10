from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sklearn


class PCA_Variants2Gene_FeatureSelection(sklearn.base.TransformerMixin):
    def __init__(self, variants_genes_path="../../data/interim/variants_top56_genes.csv",
                 variance_threshold=0.9) -> None:
        self.variants_top56_genes_path = variants_genes_path
        self.variance_threshold = variance_threshold

        self.variants_top56_genes = pd.read_table(self.variants_top56_genes_path, sep=",")
        self.variants_top56_genes.set_index("Variant ID", inplace=True)
        self.variants_top56_genes = self.variants_top56_genes.filter(items=["Gene(s)", "Variant ID"])
        super().__init__()

    def get_mutations_by_gene_dict(self, snps_df):
        # genes_matched_variants = snps_df.columns & self.variants_top56_genes.index
        mutations_gene_matched = snps_df.T.join(self.variants_top56_genes, how="right")
        mutations_by_gene = {}
        mutations_gene_gb = mutations_gene_matched.groupby("Gene(s)")

        for x in mutations_gene_gb.groups:
            mutations_gene_df = mutations_gene_gb.get_group(x)
            gene_name = mutations_gene_df["Gene(s)"].iloc[0]
            mutations_gene_df = mutations_gene_df.drop(columns=["Gene(s)"]).dropna(axis=0).T
            #     mutations_gene_df.to_csv("../data/processed/SNPs_by_gene/"+gene_name+".csv")
            mutations_by_gene[gene_name] = mutations_gene_df

        return mutations_by_gene


    def fit_transform(self, X, y=None, verbose=False):
        """

        :param X: SNPs DataFrame with sample id rows and SNP site columns (i.e. chromesome:location)
        :param fit_params:
        :return:
        """
        mutations_by_gene = self.get_mutations_by_gene_dict(X)
        self.PCA_by_gene = {}
        self.top_k_PC_by_gene = {}
        self.genes = mutations_by_gene.keys()

        # FIT PCAs
        for gene in mutations_by_gene.keys():
            self.PCA_by_gene[gene] = PCA()
            self.PCA_by_gene[gene].fit(mutations_by_gene[gene])
            top_k = np.argmax(np.cumsum(np.power(self.PCA_by_gene[gene].singular_values_, 2)) /
                              np.sum(np.power(self.PCA_by_gene[gene].singular_values_, 2)) > self.variance_threshold)
            self.top_k_PC_by_gene[gene] = top_k
            print(gene, "top_k", top_k, mutations_by_gene[gene].shape) if verbose else None

        # Transform X to top_k pca loadings by gene
        x_transform_list = []
        for gene in mutations_by_gene.keys():
            x_by_gene_transform = self.PCA_by_gene[gene].transform(mutations_by_gene[gene])
            x_transform_list.append(x_by_gene_transform[:, :self.top_k_PC_by_gene[gene]])

        pca_projs_concat = np.concatenate(x_transform_list,axis=1)
        print("pca_projs_concat.shape", pca_projs_concat.shape) if verbose else None

        pca_proj_names = []
        for i, gene in enumerate(mutations_by_gene.keys()):
            for n in range(self.top_k_PC_by_gene[gene]):
                pca_proj_names.append(gene + "_" + str(n))

        return pd.DataFrame(pca_projs_concat, index=X.index, columns=pca_proj_names)

    def transform(self, X, y=None):
        mutations_by_gene = self.get_mutations_by_gene_dict(X)
        x_transform_list = []

        # Transform X to top_k pca loadings by gene
        for gene in mutations_by_gene.keys():
            x_by_gene_transform = self.PCA_by_gene[gene].transform(mutations_by_gene[gene])
            x_transform_list.append(x_by_gene_transform[:, :self.top_k_PC_by_gene[gene]])

        pca_projs_concat = np.concatenate(x_transform_list,axis=1)

        pca_proj_names = []
        for i, gene in enumerate(mutations_by_gene.keys()):
            for n in range(self.top_k_PC_by_gene[gene]):
                pca_proj_names.append(gene + "_" + str(n))

        return pd.DataFrame(pca_projs_concat, index=X.index, columns=pca_proj_names)

    def get_gene_variability_score(self, X):
        X_1 = self.transform(X)
        patient_gene_proj_scores = pd.DataFrame(index=X_1.index, columns=self.genes)
        patient_gene_proj_scores.index.name = "patient_id"

        for gene in self.genes:
            gene_projs = X_1.iloc[:, X_1.columns.str.contains(gene)]
            patient_gene_proj_scores[gene] = np.power(gene_projs, 2).sum(axis=1) / gene_projs.columns.shape[0]

        return patient_gene_proj_scores


def get_top_variant_by_coef(pca_components, coef_percentile=70):
    gene_coefs_sum = np.abs(pca_components).sum(axis=0)
    coefs_percentile = np.percentile(gene_coefs_sum, coef_percentile)
    return np.where(gene_coefs_sum > coefs_percentile)

def select_top_variants(snp_data, top_k_components, coef_percentile=70):
    variants = snp_data.columns
    top_variants_idx = get_top_variant_by_coef(top_k_components, coef_percentile)

    return variants[top_variants_idx]





