We import the source code from the scGCC repo.
Apply scGCC to the clustering task:
```bash
python scGCC.py --input_h5ad_path="./Baron.h5" --epochs 200 --lr 0.001 --batch_size 128 --low_dim 256 --aug_prob 0.5 --num_cluster 14 --cluster_name 'kmeans' 
```
