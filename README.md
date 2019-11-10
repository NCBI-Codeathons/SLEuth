# STRATIFICATION OF SLE PATIENT COHORT FOR PRECISION MEDICINE

Systemic Lupus Erythematosus (SLE) is a complex polygenic autoimmune disease. Existing literature suggest that combination of genetic risk alleles and environmental factors can create an autoimmune-prone immune system which becomes dysregulated, leading to autoimmunity. SLE is clinically and genetically heterogeneous, which complicates its diagnosis, prognosis, management, and the development of effective therapeutic protocols. We have developed an extensive dataset combining genomic sequences of more than 28 SLE risk loci with extensive autoantibody profiles and clinical pathologies. 

Our goal is to develop analytical tools that can use these data to stratify SLE patients into subsets with significant differences in their genetics and clinical characteristics. The development of a genomic/phenotypic test that could predict clinical course and response to management would significantly improve outcomes and prevent complications of SLE. We believe that it is likely that this “precision medicine” approach will lead to the development of companion diagnostics that can be used to identify SLE patients that are most likely to respond positively to specific drug therapies. Since SLE clinical trials often show drug efficacy in a subset of patients but fail to be of significant benefit for the total patient cohort, we believe that effective subsetting of SLE patients based on genetic and phenotypic elements could dramatically improve the discovery of effective drug therapies for targeted patients subsets.

Team Lead: Prithvi Raj, Immunology, profiles.utsouthwestern.edu/profile/116973/prithvi-raj.html  


# SLEuth Readme
This Readme introduces a project in 2019 U-HACK MED hackathon at UT Southwestern Medical Center. 

Table of Contents

- [Background](#Background)
- [Workflow](#Workflow)
- [Run](#Run)
- [Licencing](#Licencing)

# Background
The project aims to implement an analytical tool to stratify Systemic Lupus Erythematosus (SLE) patients into subsets with significant differences based on their genetics variant information. Due to the observation in SLE clinical trials that drug efficacy is often seen in a subset of patients instead of the whole cohort, we believe that this stratification strategy could predict the potential clinical outcomes of a newly admitted SLE patient, and improve the discovery of effective drug therapies for targeted patients subsets. 


## Workflow  
### 1. Data Preprocessing  
1.1 The input data is a 1349 x 110K matrix. Each row is the record from one case and one column stands for one SNP site;  
1.2 Split the matrix according to the 53 risk loci to generate 53 matrices;   
1.3 Conduct PCA analysis to all the 53 matrics and only keep SNPs that contribute most to the 90% variance;   
1.4 Merge the 53 matrices and generate 1349 x 34K matrix.  
### 2. Data Clustering  
2.1 Conduct K-means to split the data into 6 clusters;  
2.2 Conduct spectral clustering. 
### 3. Confirmation of Difference
3.1 For each gene in a cluster, the gene score is calculated to reflect the frequency and variability of its SNPs        
<a href="https://www.codecogs.com/eqnedit.php?latex=SC&space;=&space;\frac{\sum_(N&space;of&space;readings,&space;SNP_i)^2}{\sum&space;SNPs}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SC&space;=&space;\frac{\sum_(N&space;of&space;readings,&space;SNP_i)^2}{\sum&space;SNP_i}" title="SC = \frac{\sum_(N of readings, SNP_i)^2}{\sum SNPs}" /></a> where the sum of SNP_i means the total number of different SNPs in a gene;    
3.2 The similarity levels of each gene in patients are calculated.
# Run  
Download the dataset "" and run:
```Python
