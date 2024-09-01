library(scImpute)

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


count_path <-'./Baron_Droup0.1_x.csv'
out_dir <- './'

pbmc <- readRDS('./Baron.rds')
unique_clusters <- unique(pbmc@meta.data$true_cluster)
num_unique_clusters <- length(unique_clusters)
labels_true <- pbmc@meta.data$true_cluster

scimpute(count_path=count_path, infile = "csv", outfile = "csv", type = "count",
         out_dir=out_dir, labeled = FALSE, drop_thre = 0.5, Kcluster = num_unique_clusters,
         labels = NULL, genelen = NULL, ncores = 1)