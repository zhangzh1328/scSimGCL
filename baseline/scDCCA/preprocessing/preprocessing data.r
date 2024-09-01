scRNA<- readRDS("muraro.rds")


#############without log#############
# counts<-scRNA@assays[[".->data"]]@listData[["counts"]]
# counts<-as.matrix(counts)
# dim(counts)
# sce<-counts
# #counts(sce)
# #ncol(sce)
# #rnorm(ncol(sce))
# sf <- 2^rnorm(ncol(sce))
# #sf
# sf <- sf/mean(sf)
# normcounts<- t(t(sce)/sf)
# dim(normcounts)
# class(normcounts)
# logcounts <- log2(normcounts+1)
# #View(logcounts)
#########with log####################
logcounts<-scRNA@assays[[".->data"]]@listData[["logcounts"]]
########################
class(logcounts)
dim(logcounts)
labels1<-scRNA@colData@listData[["cell_type1"]]
labels2<-scRNA@colData@listData[["cell_type2"]]
class(labels1)
dim(labels1)
######### converts into boolean matrix ###
logcounts.dataframe<-as.data.frame(logcounts)
labels1.dataframe<-as.data.frame(labels1)
labels2.dataframe<-as.data.frame(labels2)

logcounts.dataframe_beifen<-logcounts.dataframe
class(logcounts.dataframe)
logcounts.dataframe[logcounts.dataframe > 0] = 1
dim(logcounts.dataframe)
######### filtering: gene filtering ########
rowdata.sum = rowSums(logcounts.dataframe)
coldata.sum = colSums(logcounts.dataframe)
length(rowdata.sum) 
length(coldata.sum) 
#View(rowdata.sum)

logcounts.dataframe.genefilt <- logcounts.dataframe_beifen[which(rowdata.sum>length(coldata.sum)*0.05),]

dim(logcounts.dataframe.genefilt) 
logcounts.dataframe.genefilt.t<-t(logcounts.dataframe.genefilt)
dim(logcounts.dataframe.genefilt.t)
#########################################
write.table(logcounts.dataframe.genefilt.t,"muraro_preprocessed_log_genefilt.csv",row.names=TRUE,col.names=TRUE,sep=" ")
write.table(labels1,"muraro_truelabels1.csv",row.names=FALSE,col.names=FALSE,sep=" ")

write.table(logcounts.dataframe.genefilt.t,"muraro_preprocessed_log_genefilt.txt",row.names=FALSE,col.names=FALSE,sep=" ")
write.table(labels1,"muraro_truelabels.txt",row.names=FALSE,col.names=FALSE,sep=" ")

