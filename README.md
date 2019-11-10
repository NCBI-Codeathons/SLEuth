# STRATIFICATION OF SLE PATIENT COHORT FOR PRECISION MEDICINE

Systemic Lupus Erythematosus (SLE) is a complex polygenic autoimmune disease. Existing literature suggest that combination of genetic risk alleles and environmental factors can create an autoimmune-prone immune system which becomes dysregulated, leading to autoimmunity. SLE is clinically and genetically heterogeneous, which complicates its diagnosis, prognosis, management, and the development of effective therapeutic protocols. We have developed an extensive dataset combining genomic sequences of more than 28 SLE risk loci with extensive autoantibody profiles and clinical pathologies. 

Our goal is to develop analytical tools that can use these data to stratify SLE patients into subsets with significant differences in their genetics and clinical characteristics. The development of a genomic/phenotypic test that could predict clinical course and response to management would significantly improve outcomes and prevent complications of SLE. We believe that it is likely that this “precision medicine” approach will lead to the development of companion diagnostics that can be used to identify SLE patients that are most likely to respond positively to specific drug therapies. Since SLE clinical trials often show drug efficacy in a subset of patients but fail to be of significant benefit for the total patient cohort, we believe that effective subsetting of SLE patients based on genetic and phenotypic elements could dramatically improve the discovery of effective drug therapies for targeted patients subsets.

Team Lead: Prithvi Raj, Immunology, profiles.utsouthwestern.edu/profile/116973/prithvi-raj.html  


# SLEuth Readme
This Readme introduces a project in 2019 U-HACK MED hackathon at UT Southwestern Medical Center. 

Table of Contents

- [Background](#Background)
- [Workflow](#Workflow)
- [Licencing](#Licencing)  


# Background
The project aims to implement an analytical tool to stratify Systemic Lupus Erythematosus (SLE) patients into subsets with significant differences based on their genetics variant information. Due to the observation in SLE clinical trials that drug efficacy is often seen in a subset of patients instead of the whole cohort, we believe that this stratification strategy could predict the potential clinical outcomes of a newly admitted SLE patient, and improve the discovery of effective drug therapies for targeted patients subsets. 


# Workflow  
The data consisted of approximately 110,000 SNPs for 630 SLE patients at 53 loci of interest identified by previous studies. The SNPs were split according to loci. At each locus the SNPs were independently reduced in dimension using PCA. The reduced coordinates were then re-aggregated to form a matrix of 630 patients by 1386 PCA coordinates. These data were clustered by three independent methods: , 

### 1. Data Preprocessing  
1.1 The input data is a 630 x 110K matrix. Each row is the record from one case and one column stands for one SNP site;  
1.2 Split the matrix according to the 53 risk loci to generate 53 matrices;   
1.3 Conduct PCA analysis to all the 53 matrics and only keep SNPs that contribute most to the 90% variance;   
1.4 Merge the 53 matrices and generate 630 x 1386 matrix.  
### 2. Data Clustering  

Three methods were implemented and compared: 

1. K-means clustering
1. Spectral clustering (nearest-neighbor and Gaussian similarity kernels)
1. Hierarchical clustering

Quality of clusters was assessed by silhouette plots and the overall silhouette score. All processing took place in the scikit-learn module version 0.21.3. 

The nearest-neighbor kernel was found to give inferior results to other clustering methods and was discarded. The remaining three clustering methods gave clusters of similar sizes and quality. K-means clusters were used for final results. UMAP projection was then used to visualize the clusters. 

### 3. Classifying New Data
Once clusters have been found, data for new patients can be projected to the PCA coordinates and compared to each of the cluster centroids. New data can also be visualized using the UMAP projection to get visual information about the proximity to the clusters.   

# Licencing  
Python >=3.6, pandas>=0.24.2, numpy >= 1.16.4, scipy >= 1.2.1, matplotlib >= 3.1.0, scikit-learn >= 0.21.3, seaborn >= 0.9.0, jupyter-notebook >= 6.0.1, umap-learn >= 0.3.10

