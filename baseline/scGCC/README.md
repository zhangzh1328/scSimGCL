# Run scGCC
python scGCC.py --input_h5ad_path="/data1/zzh/Bioinfo/esm-main/single_cell/single_cell/data/Bioinfo-scGPCL-DataSet/Baron.h5" --epochs 100 --lr 0.001 --batch_size 128 --low_dim 256 --aug_prob 0.5 --num_cluster 14 --cluster_name 'kmeans' 
