library(cidr)
library(hdf5r)
library(Seurat)
library(ggplot2)
library(cowplot)
library(Matrix)
library(dplyr)
library(ggsci)
library(scater)
library(SingleR)
library(Matrix)
library(flexclust)
library(aricode)

gene = read.table('./Baron_geneinfo.csv',stringsAsFactors=F, header=T,sep = ',')
gene = gene$gene_id
gene = make.unique(gene, sep = '.')
cell = read.table('./Baron_cellinfo.csv',stringsAsFactors=F, header=T,sep = ',')
barcodes=cell$cell_id
mtx <- read.table('./Baron_matrix.csv',stringsAsFactors=F,header=T,sep = ' ')
mtx <- as.matrix(mtx)
mtx[is.na(mtx)] <- 0

colnames(mtx) = barcodes
rownames(mtx) = gene

scBrain <- scDataConstructor(as.matrix(mtx))
scBrain <- determineDropoutCandidates(scBrain)
scBrain <- wThreshold(scBrain)
scBrain <- scDissim(scBrain)
scBrain <- scPCA(scBrain)
scBrain <- nPC(scBrain)
nCluster(scBrain)
scBrain <- scCluster(scBrain)
labels_pred <- scBrain@clusters
pred <- append(pred, list(labels_pred))