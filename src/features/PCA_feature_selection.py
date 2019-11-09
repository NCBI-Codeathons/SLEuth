from sklearn.decomposition import PCA
import numpy as np


def get_top_k_components(snp_data, var_threshold=0.80, return_fit_transform=False):
    pca = PCA()
    pca.fit(snp_data)
    top_k = np.argmax(np.cumsum(pca.singular_values_) / np.sum(pca.singular_values_) > var_threshold)

    if return_fit_transform:
        pca.n_components = top_k
        pca.n_components_ = top_k
        return pca.transform(snp_data)[:, :top_k]
    else:
        return pca.components_[:top_k]

def get_top_variant_by_coef(pca_components, coef_percentile=70):
    gene_coefs_sum = np.abs(pca_components).sum(axis=0)
    coefs_percentile = np.percentile(gene_coefs_sum, coef_percentile)
    return np.where(gene_coefs_sum > coefs_percentile)

def select_top_variants(snp_data, var_threshold=0.80, coef_percentile=70):
    variants = snp_data.columns
    top_k_components = get_top_k_components(snp_data, var_threshold)
    top_variants_idx = get_top_variant_by_coef(top_k_components, coef_percentile)

    return variants[top_variants_idx]





