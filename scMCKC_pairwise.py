from time import time
import math, os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scMCKC import scMCKC
import numpy as np
import collections
from sklearn import metrics
import h5py
import scanpy as sc
from preprocess import *
from utils import cluster_acc, generate_random_pair_from_proteins, generate_random_pair_from_CD_markers, normalizeSC
import pandas as pd
import scanpy as sc
import scipy as sp
import numpy as np



if __name__ == "__main__":

    # setting the hyper parameters
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_clusters', default=8, type=int)
    parser.add_argument('--n_pairwise_1', default=0, type=int)
    parser.add_argument('--n_pairwise_2', default=0, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--data_file', default='./data/baseline_Datasets/Adam.h5')
    parser.add_argument('--maxiter', default=2000, type=int)
    parser.add_argument('--pretrain_epochs', default=300, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of clustering loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--tol', default=0.001, type=float)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--paras', default='default_')
    parser.add_argument('--save_dir', default='results/Adam/')
    parser.add_argument('--ae_weight_file', default='pretrained_weights/AE_weights_p0_1.pth.tar')
    parser.add_argument('--latent_z', default='latent_p0_1.txt')
    

    args = parser.parse_args()

    X, Y, batch_label = prepro(args.data_file)
    label_vec = torch.tensor(Y.astype(np.float32))
    X = np.ceil(X).astype(np.int)
    count_X = X
    cellname = np.array(["group" + str(i) for i in Y])
    print("cellname:\n", cellname)
    highly_genes = 500


    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata.obs["celltype"] = cellname
    adata.obs["batch"] = batch_label
    adata = normalizeSC(adata, copy=True, highly_genes=500, size_factors=True, normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)

    embedding = X
    y = np.array(adata.obs["Group"])
    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
    count_X = count_X[:, high_variable]

    cellname = np.array(adata.obs["celltype"])
    batch_label = np.array(adata.obs["batch"])

    cluster_num = len(np.unique(cellname))
    batch_num = len(np.unique(batch_label))

    markers = np.loadtxt("./pretrained_weights/adt_CD_normalized_counts.txt", delimiter=',')

    input_size = adata.n_vars
    print(args)

    print(adata.X.shape)
    print(y.shape)

    x_sd = adata.X.std(0)
    x_sd_median = np.median(x_sd)
    print("median of gene sd: %.5f" % x_sd_median)

    if args.n_pairwise_1 > 0:
        ml_ind1_1, ml_ind2_1, cl_ind1_1, cl_ind2_1 = generate_random_pair_from_proteins(embedding, args.n_pairwise_1, 0.005, 0.95)

        print("Must link paris: %d" % ml_ind1_1.shape[0])
        print("Cannot link paris: %d" % cl_ind1_1.shape[0])
    else:
        ml_ind1_1, ml_ind2_1, cl_ind1_1, cl_ind2_1 = np.array([]), np.array([]), np.array([]), np.array([])

    if args.n_pairwise_2 > 0:
        ml_ind1_2, ml_ind2_2, cl_ind1_2, cl_ind2_2 = generate_random_pair_from_CD_markers(markers, args.n_pairwise_2, 0.3, 0.7, 0.3, 0.85)

        print("Must link paris: %d" % ml_ind1_2.shape[0])
        print("Cannot link paris: %d" % cl_ind1_2.shape[0])
    else:
        ml_ind1_2, ml_ind2_2, cl_ind1_2, cl_ind2_2 = np.array([]), np.array([]), np.array([]), np.array([])

    ml_ind1 = np.append(ml_ind1_1, ml_ind1_2)
    ml_ind2 = np.append(ml_ind2_1, ml_ind2_2)
    cl_ind1 = np.append(cl_ind1_1, cl_ind1_2)
    cl_ind2 = np.append(cl_ind2_1, cl_ind2_2)

    sd = 2.5

    model = scMCKC(input_dim=adata.n_vars, z_dim=32, n_clusters=args.n_clusters,
                encodeLayer=[256, 64], decodeLayer=[64, 256], sigma=sd, gamma=args.gamma, batch_label=batch_label, label_vec=label_vec).cuda()

    print(str(model))

    t0 = time()
    if args.ae_weights is None:
        model.pretrain_autoencoder(x=adata.X, X_raw=count_X, size_factor=adata.obs.size_factors,
                                batch_size=args.batch_size, epochs=args.pretrain_epochs, ae_weights=args.ae_weight_file)
    else:
        if os.path.isfile(args.ae_weights):
            print("==> loading checkpoint '{}'".format(args.ae_weights))
            checkpoint = torch.load(args.ae_weights)
            model.load_state_dict(checkpoint['ae_state_dict'])
        else:
            print("==> no checkpoint found at '{}'".format(args.ae_weights))
            raise ValueError
    
    print('Pretraining time: %d seconds.' % int(time() - t0))

    if not os.path.exists(args.save_dir+args.paras):
            os.makedirs(args.save_dir+args.paras)

    y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=count_X, sf=adata.obs.size_factors, y=y, batch_size=args.batch_size, num_epochs=args.maxiter,
                ml_ind1=ml_ind1, ml_ind2=ml_ind2, cl_ind1=cl_ind1, cl_ind2=cl_ind2,
                update_interval=args.update_interval, tol=args.tol, save_dir=args.save_dir+args.paras) #
    print('Total time: %d seconds.' % int(time() - t0))

    acc = np.round(cluster_acc(y, y_pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Evaluating cells: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

    latent_z0 = model.encodeBatch(torch.tensor(adata.X).cuda())
    latent_z = latent_z0.data.cpu().numpy()
    np.savetxt(args.save_dir+args.paras+args.latent_z, latent_z, delimiter=",")
    np.savetxt(args.save_dir+args.paras+'pred_y_'+args.latent_z, np.array(y_pred), delimiter=",")
    print('Total time: %d seconds.' % int(time() - t0))

    result_str = 'ACC=%.4f NMI=%.4f ARI=%.4f' % (acc, nmi, ari)
    print(result_str)
    fh = open(args.save_dir + args.paras + result_str +'.txt', 'w', encoding='utf-8')
    fh.write(result_str)
    fh.close()