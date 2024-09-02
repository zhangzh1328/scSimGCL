We import the source code from the scGNN repo.  
Apply scGNN to the clustering task:
```bash
python -W ignore scGNN.py --datasetDir './'  --outputDir ./outputdir/  --clustering-method 'KMeans' --batch-size 512 --Regu-epochs 200 --EM-epochs 200 --seed 1 --GAEepochs 200 --datasetName 'Baron' --EM-iteration 10 --n-clusters 14 --nonsparseMode --quickmode
```
Apply scGNN to the imputation task: 
```bash
python -W ignore scGNN_Imp.py --datasetDir './'  --outputDir ./outputdir/ --nonsparseMode --clustering-method 'KMeans' --batch-size 512 --Regu-epochs 200 --EM-epochs 200 --seed 1 --GAEepochs 200  --n-clusters 14 --datasetName 'Baron.h5' --my_drop_rate 0.1
```
