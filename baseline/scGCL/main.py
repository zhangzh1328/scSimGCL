import torch

# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)
import data_Preprocess
import numpy as np
np.random.seed(0)
import utils
import pandas as pd

def main():
    args, unknown = utils.parse_args()

    if args.embedder == 'AFGRL':
        from models import AFGRL_ModelTrainer
        embedder = AFGRL_ModelTrainer(args)

    embedder.train()
    embedder.writer.close()

def imputation(file_path):
    args, unknown = utils.parse_args()
    imputation_m = torch.load(file_path).detach().data.cpu().numpy()
    a = pd.DataFrame(imputation_m).T
    a.to_csv("./results/AFGRL-imputed-" + args.dataset + ".csv")



if __name__ == "__main__":
    main()
    #imputation("./model_checkpoints/embeddings_Adam_clustering.pt")

