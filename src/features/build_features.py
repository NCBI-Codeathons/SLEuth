import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

vcf = allel.read_vcf("../../data/raw/1349 sample and all 253k unfiltered SNPs.vcf", )
variants = np.char.array(vcf["variants/CHROM"].astype(str)) + ":" + np.char.array(vcf["variants/POS"].astype(str))
vcf_arr = vcf["calldata/GT"].astype("float")
vcf_arr[vcf_arr == -1] = np.nan

mutations = vcf_arr
# mutations = np.abs(mutations)
mutations = mutations.sum(axis=2)
mutations = mutations.T

mutations_df = pd.DataFrame(data=mutations, index=vcf["samples"], columns=variants)
mutations_df.dropna(axis=1, how="any", thresh=800, inplace=True)
mutations_df.dropna(axis=0, how="any", thresh=200000, inplace=True)
mutations_df.fillna(value=0, inplace=True)

# Subset patients
samples_phenotypes = pd.read_table("../../data/raw/Sample metadata.csv", sep=",")
samples_phenotypes.set_index("ID", inplace=True)
good_samples = pd.read_table("../../data/interim/samples_metadata.csv", sep=",")
good_samples.set_index("ID", inplace=True)
good_samples = good_samples[good_samples["SRC"] != "LGS"]
good_samples = good_samples[good_samples["SRC"] != "D2"]
good_samples = good_samples[good_samples["SRC"] != "U2"]

SLE_samples = good_samples[good_samples["SLE"] == 1]

hla_protein_samples = pd.Index(['55062', '56104', '34903', '16820', '41060', '54687', '44119', '48523',
       '33287', '14947', '21560', '87483', '42335', '30146', '28289', '40007'])

highdsdna_samples = pd.Index(["32588", "55062"]) # High dsDNA
lowdsdna_samples = pd.Index(["54687", "16820"]) # low dsDNA
validation_samples = highdsdna_samples.append(lowdsdna_samples)
training_samples = SLE_samples.index[~SLE_samples.index.isin(validation_samples)]

# Filter data for
training_df = mutations_df.filter(items=training_samples, axis=0)
# training_df.shape

validation_df = mutations_df.filter(items=validation_samples, axis=0)
# validation_df.shape

from .feature_selection import PCA_Variants2Gene_FeatureSelection

pca_v2g = PCA_Variants2Gene_FeatureSelection(variants_genes_path="../../data/interim/variants_top56_genes.csv",
                 variance_threshold=0.80)
training_X1 = pca_v2g.fit_transform(training_df)
training_X1.index.name = "patient_id"
validation_X1 = pca_v2g.transform(validation_df)
validation_X1.index.name = "patient_id"
# training_X1.to_csv("data/processed/training_pca_projs.csv")
# validation_X1.to_csv("data/processed/validation_pca_projs.csv")

training_gene_scores = pca_v2g.get_gene_variability_score(pd.concat([training_df, validation_df], axis=0))
training_gene_scores
training_gene_scores.to_csv("../../data/processed/allsamples_gene_variability_scores.csv")