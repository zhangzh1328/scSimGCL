# scDCCA
We present a new deep contrastive clustering algorithm (scDCCA). scDCCA extracts valuable features and realizes cell segregation end-to-end by introducing dual contrastive learning and denoising ZINB model-based auto-encoder into a deep clustering framework. The results validate that ScDCCA outperforms eight state-of-the-art methods in terms of accuracy, generalizability, scalability, and efficiency. 

See details in our paper: "scDCCA: Deep contrastive clustering for single-cell RNA-seq data based on auto-encoder network". 

Quick start:  
Download the datasets from the website as the paper described, and download the code files in this repository and import functions in them.  
1.Preprocess datasets  
scDCCA takes preprocessed data as input. Single cell data preprocessing can be done with preprocessing data.r and select 2000 highly genes.py.  
2.Run scDCCA algorithm with mymodel.py. 
3.Biological analysis
  Perform biological analysis with cosg_test.py.

Requirements: 

python---3.7 
scanpy---1.7.2
numpy --- 1.21.2  
pandas --- 1.3.4
cosg---1.0.1
scikit-learn --- 1.0.2
matplotlib --- 3.5.0
