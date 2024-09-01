library(SAVER)


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

data <- read.csv('./Baron_Droup0.1_x.csv',header = T,row.names = 1)
cortex <- as.matrix(data)
cortex.saver <- saver(cortex,ncores = 1)

write.csv(cortex.saver[["estimate"]], './Baron_SAVER_Droup0.1_x.csv', row.names = TRUE)