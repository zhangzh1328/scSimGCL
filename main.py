import argparse
import warnings
import torch
import numpy as np
from utils import setup_seed, loader_construction, evaluate, device
from scSimGCL import Model
from sklearn.cluster import KMeans
from config import config


def train(train_loader,
          test_loader,
          input_dim,
          graph_head,
          phi,
          gcn_dim,
          mlp_dim,
          prob_feature,
          prob_edge,
          tau,
          alpha,
          beta,
          lambda_cl,
          dropout,
          lr,
          seed,
          epochs,
          save_model_path,
          device):

    model = Model(input_dim=input_dim, graph_head=graph_head, phi=phi, gcn_dim=gcn_dim,
                  mlp_dim=mlp_dim, prob_feature=prob_feature, prob_edge=prob_edge, tau=tau,
                  alpha=alpha, beta=beta, dropout=dropout).to(device)

    opt_model = torch.optim.Adam(model.parameters(), lr=lr)

    setup_seed(seed)
    test_loss = []
    best_epoch = 0
    min_loss = 999

    np.set_printoptions(threshold=np.inf)
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)

    z_train_epoch = []
    z_test_epoch = []
    y_train_epoch = []
    y_test_epoch = []
    x_imp_train_epoch = []
    x_imp_test_epoch = []

    for each_epoch in range(epochs):
        z_train = []
        z_test = []
        y_train = []
        y_test = []
        x_imp_train = []
        x_imp_test = []

        batch_loss = []
        model.train()

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)

            batch_z, x_imp, loss_cl = model(batch_x)
            mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                               torch.zeros(batch_x.shape).to(device))
            mae_f = torch.nn.L1Loss(reduction='mean')
            loss_mae = mae_f(mask * x_imp, mask * batch_x)

            loss = loss_mae + lambda_cl * loss_cl

            z_train.append(batch_z.cpu().detach().numpy())
            y_train.append(batch_y)
            x_imp_train.append(x_imp.cpu().detach().numpy())

            opt_model.zero_grad()
            loss.backward()
            opt_model.step()

            batch_loss.append(loss.cpu().detach().numpy())

        with torch.no_grad():
            batch_loss = []
            model.eval()

            for step, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)

                batch_z, x_imp, loss_cl = model(batch_x)
                mask = torch.where(batch_x != 0, torch.ones(batch_x.shape).to(device),
                                   torch.zeros(batch_x.shape).to(device))
                loss_mae = mae_f(mask * x_imp, mask * batch_x)

                loss = loss_mae + lambda_cl * loss_cl

                z_test.append(batch_z.cpu().detach().numpy())
                y_test.append(batch_y)
                x_imp_test.append(x_imp.cpu().detach().numpy())

                batch_loss.append(loss.cpu().detach().numpy())

        test_loss.append(np.mean(np.array(batch_loss)))
        cur_loss = test_loss[-1]

        if cur_loss < min_loss:
            min_loss = cur_loss
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': opt_model.state_dict(),
                'epoch': each_epoch
            }
            torch.save(state, save_model_path)

        z_train_epoch.append(z_train)
        z_test_epoch.append(z_test)
        y_train_epoch.append(y_train)
        y_test_epoch.append(y_test)
        x_imp_train_epoch.append(x_imp_train)
        x_imp_test_epoch.append(x_imp_test)

    return best_epoch, min_loss, z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch, x_imp_train_epoch, x_imp_test_epoch


def test(z_test_epoch,
         y_test_epoch,
         best_epoch,
         n_clusters,
         seed):

    z_test = z_test_epoch[best_epoch]
    y_test = y_test_epoch[best_epoch]

    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_kmeans_test = kmeans.labels_

    acc, f1, nmi, ari, homo, comp = evaluate(y_test, y_kmeans_test)
    results = {'CA': acc, 'NMI': nmi, 'ARI':ari}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_head", type=int, default=config['graph_head'])
    parser.add_argument("--phi", type=float, default=config['phi'])
    parser.add_argument("--gcn_dim", type=int, default=config['gcn_dim'])
    parser.add_argument("--mlp_dim", type=int, default=config['mlp_dim'])
    parser.add_argument("--prob_feature", type=float, default=config['prob_feature'])
    parser.add_argument("--prob_edge", type=float, default=config['prob_edge'])
    parser.add_argument("--tau", type=float, default=config['tau'])
    parser.add_argument("--alpha", type=float, default=config['alpha'])
    parser.add_argument("--beta", type=float, default=config['beta'])
    parser.add_argument("--lambda_cl", type=float, default=config['lambda_cl'])
    parser.add_argument("--dropout", type=float, default=config['dropout'])
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--lr", type=float, default=config['lr'])
    parser.add_argument("--seed", type=int, default=config['seed'])
    parser.add_argument("--epochs", type=int, default=config['epochs'])
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--save_model_path", type=str)
    args = parser.parse_args()

    graph_head = args.graph_head
    phi = args.phi
    gcn_dim = args.gcn_dim
    mlp_dim = args.mlp_dim
    prob_feature = args.prob_feature
    prob_edge = args.prob_edge
    tau = args.tau
    alpha = args.alpha
    beta = args.beta
    lambda_cl = args.lambda_cl
    dropout = args.dropout
    n_clusters = args.n_clusters
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    data_path = args.data_path
    save_model_path = args.save_model_path

    train_loader, test_loader, input_dim = loader_construction(data_path)
    best_epoch, min_loss, z_train_epoch, z_test_epoch, y_train_epoch, y_test_epoch, x_imp_train_epoch, x_imp_test_epoch = train(train_loader, test_loader, input_dim, graph_head, phi, gcn_dim, mlp_dim, prob_feature, prob_edge,
                       tau, alpha, beta, lambda_cl, dropout, lr, seed, epochs, save_model_path, device)
    results = test(z_test_epoch, y_test_epoch, best_epoch, n_clusters, seed)
    print(results)
