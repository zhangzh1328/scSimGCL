library(igraph)
library(SIMLR)

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

mtx <- read.csv("./Baron_matrix.csv",header = TRUE, row.names = 1)
unique_clusters <- unique(pbmc@meta.data$true_cluster)
num_unique_clusters <- length(unique_clusters)
 
example = SIMLR(X = mtx, c = num_unique_clusters, cores.ratio = 0)
labels_pred <- example$y$cluster
