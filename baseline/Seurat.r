library(hdf5r)
library(Seurat)
library(ggplot2)
library(cowplot)
library(Matrix)
library(dplyr)
library(ggsci)
library(scater)

library(Matrix)
library(flexclust)
library(aricode)

pbmc <- readRDS('./Baron.rds')
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
unique_clusters <- unique(pbmc@meta.data$true_cluster)
num_unique_clusters <- length(unique_clusters)
  
k.param = num_unique_clusters
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = num) 
pbmc <- RunUMAP(pbmc, dims = 1:10, label = T)

labels_true <- pbmc@meta.data$true_cluster
labels_pred <- pbmc@meta.data$seurat_clusters