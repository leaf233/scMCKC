import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from layers import ZINBLoss, MeanAct, DispAct
import numpy as np
from sklearn.cluster import KMeans
import math, os
from sklearn import metrics
from utils import cluster_acc
import random
from scipy.linalg import norm
import pandas as pd
import scanpy as sc


def buildNetwork(layers, type, activation="relu"):
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i - 1], layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
    return nn.Sequential(*net)


class scMCKC(nn.Module):
    def __init__(self, input_dim, z_dim, n_clusters, batch_label, label_vec, encodeLayer=[], decodeLayer=[],
                 activation="relu", sigma=1., alpha=1., gamma=1., beta=1., alpha_zinb=1.,
                 ml_weight=1., cl_weight=1., cell_weight=1.,
                 ):
        super(scMCKC, self).__init__()
        self.z_dim = z_dim
        self.n_clusters = n_clusters
        self.activation = activation
        self.sigma = sigma
        self.alpha = alpha
        self.alpha_zinb = alpha_zinb
        self.gamma = gamma
        self.beta = beta
        self.ml_weight = ml_weight
        self.cl_weight = cl_weight
        self.cell_weight = cell_weight
        self.encoder = buildNetwork([input_dim] + encodeLayer, type="encode", activation=activation)
        self.decoder = buildNetwork([z_dim] + decodeLayer, type="decode", activation=activation)
        self._enc_mu = nn.Linear(encodeLayer[-1], z_dim)
        self._dec_mean = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(decodeLayer[-1], input_dim), nn.Sigmoid())

        self.mu = Parameter(torch.Tensor(n_clusters, z_dim))
        self.zinb_loss = ZINBLoss().cuda()
        self.batch_label = batch_label
        self.label_vec = label_vec

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()

        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)

        return (p.t() / p.sum(1)).t()

    def cal_dist(self, z, clusters):
        dist1 = torch.sum(torch.square(torch.unsqueeze(z, dim=1) - clusters), dim=2)
        temp_dist1 = dist1 - torch.reshape(torch.mean(dist1, dim=1), [-1, 1])
        p = torch.exp(-temp_dist1)
        p = (p.t() / torch.sum(p, dim=1)).t()
        p = p ** 2
        p = (p.t() / torch.sum(p, dim=1)).t()
        dist2 = dist1 * p
        return dist1, dist2

    def forward(self, x):
        h = self.encoder(x + torch.randn_like(x) * self.sigma)
        z = self._enc_mu(h)
        h = self.decoder(z)
        _mean = self._dec_mean(h)
        _disp = self._dec_disp(h)
        _pi = self._dec_pi(h)

        h0 = self.encoder(x)
        z0 = self._enc_mu(h0)
        q = self.soft_assign(z0)
        batch_label = self.batch_label
        label_vec = self.label_vec
        return z0, q, _mean, _disp, _pi, batch_label, label_vec

    def encodeBatch(self, X, batch_size=256):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()

        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
            inputs = Variable(xbatch)
            z, _, _, _, _, _, _ = self.forward(inputs)
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)
        return self.gamma * kldloss

    def kmeans_loss(self, latent_dist2):
        kloss = torch.mean(torch.sum(latent_dist2, dim=1))
        return self.beta * kloss

    def pairwise_loss(self, p1, p2, cons_type):
        if cons_type == "ML":
            ml_loss = torch.mean(-torch.log(torch.sum(p1 * p2, dim=1)))
            return self.ml_weight * ml_loss
        else:
            cl_loss = torch.mean(-torch.log(1.0 - torch.sum(p1 * p2, dim=1)))
            return self.cl_weight * cl_loss

    def pretrain_autoencoder(self, x, X_raw, size_factor, batch_size=256, lr=0.001, epochs=300, ae_save=True,
                             ae_weights='AE_weights.pth.tar', *args, **kwargs):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Pretraining stage")
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epochs):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch).cuda()
                x_raw_tensor = Variable(x_raw_batch).cuda()
                sf_tensor = Variable(sf_batch).cuda()
                _, _, mean_tensor, disp_tensor, pi_tensor, _, _ = self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor,
                                      scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx + 1, epoch + 1, loss.item()))

        if ae_save:
            torch.save({'ae_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, ae_weights)

    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)

    def fit(self, X, X_raw, sf,
            ml_ind1=np.array([]), ml_ind2=np.array([]), cl_ind1=np.array([]), cl_ind2=np.array([]),
            ml_p=1., cl_p=1.,
            y=None, lr=1., batch_size=256, num_epochs=10, update_interval=1, tol=1e-3, save_dir=""
            ):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        print("Clustering stage")
        X = torch.tensor(X).cuda()
        X_raw = torch.tensor(X_raw).cuda()
        sf = torch.tensor(sf).cuda()
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)

        print("Initializing cluster centers with kmeans.")
        kmeans = KMeans(n_clusters=self.n_clusters, init="k-means++", n_init=20)
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.cpu().numpy())
        self.y_pred_last = self.y_pred
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        if y is not None:
            acc = np.round(cluster_acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))

        self.train()
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))
        ml_num_batch = int(math.ceil(1.0 * ml_ind1.shape[0] / batch_size))
        cl_num_batch = int(math.ceil(1.0 * cl_ind1.shape[0] / batch_size))
        cl_num = cl_ind1.shape[0]
        ml_num = ml_ind1.shape[0]

        final_acc, final_nmi, final_ari, final_epoch = 0, 0, 0, 0
        update_ml = 1
        update_cl = 1

        fh = open(save_dir + 'Loss' + '.txt', 'w', encoding='utf-8')

        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                latent = self.encodeBatch(X)
                q = self.soft_assign(latent)
                p = self.target_distribution(q).data
                latent_dist1, latent_dist2 = self.cal_dist(latent, self.n_clusters)
                self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()


                if y is not None:
                    final_acc = acc = np.round(cluster_acc(y, self.y_pred), 5)
                    final_nmi = nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
                    final_epoch = ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                    print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch + 1, acc, nmi, ari))

                # save current model
                '''
                if (epoch > 0 and delta_label < tol) or epoch % 10 == 0:
                    self.save_checkpoint({'epoch': epoch + 1,
                                          'state_dict': self.state_dict(),
                                          'mu': self.mu,
                                          'p': p,
                                          'q': q,
                                          'y_pred': self.y_pred,
                                          'y_pred_last': self.y_pred_last,
                                          'y': y
                                          }, epoch + 1, filename=save_dir)
                '''
                # check stop criterion
                delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
                self.y_pred_last = self.y_pred
                if epoch > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print("Reach tolerance threshold. Stopping training.")
                    break

            train_loss = 0.0
            recon_loss_val = 0.0
            cluster_loss_val = 0.0
            kmeans_loss_val = 0.0
            similarity_loss_val = 0.0

            for batch_idx in range(num_batch):
                xbatch = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                xrawbatch = X_raw[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                sfbatch = sf[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                pbatch = p[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]
                optimizer.zero_grad()
                inputs = Variable(xbatch)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
                target = Variable(pbatch)

                z, qbatch, meanbatch, dispbatch, pibatch, batch_label, label_vec = self.forward(inputs)

                label_vec = label_vec[batch_idx * batch_size: min((batch_idx + 1) * batch_size,
                                                                  num)]
                batch_label_new = batch_label[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]

                mask_vec = pd.to_numeric(batch_label_new, errors='coerce') - 1 + 1
                mask_vec[mask_vec != 0] = 1
                mask_vec = torch.tensor(1 - mask_vec)
                label_mat = torch.tensor(torch.reshape(label_vec, [-1, 1]) - torch.reshape(label_vec, [1, -1]))
                label_mat = torch.tensor(torch.equal(label_mat, torch.zeros(len(label_mat))), dtype=torch.float64)
                label_mat = label_mat.cuda()

                mask_mat = torch.matmul(torch.reshape(mask_vec, [-1, 1]),
                                        torch.reshape(mask_vec, [1, -1]))

                mask_mat = mask_mat.cuda()
                normalize_latent = torch.nn.functional.normalize(z, dim=1)
                similarity = torch.tensor(torch.matmul(normalize_latent, normalize_latent.t()))
                similarity = similarity.cuda()
                cross_entropy = torch.tensor(mask_mat * (-label_mat * torch.log(torch.clamp(similarity, 1e-10, 1.0)) -
                                                         (1 - label_mat) * torch.log(
                                                             torch.clamp(1 - similarity, 1e-10, 1.0))))
                cross_entropy = torch.sum(cross_entropy)

                cluster_loss = self.cluster_loss(target, qbatch)
                recon_loss = self.zinb_loss(rawinputs, meanbatch, dispbatch, pibatch, sfinputs) * self.alpha_zinb
                kmeans_loss = self.kmeans_loss(latent_dist2)
                similarity_loss = cross_entropy
                loss = cluster_loss + recon_loss + kmeans_loss * 0.00015 + similarity_loss * self.cell_weight* 10000
                loss.backward()
                optimizer.step()
                cluster_loss_val += cluster_loss.data * len(inputs)
                recon_loss_val += recon_loss.data * len(inputs)
                kmeans_loss_val += kmeans_loss.data * len(inputs) * 0.00015
                similarity_loss_val += similarity_loss.data * len(inputs)* 10000
                train_loss = cluster_loss_val + recon_loss_val + kmeans_loss_val * 0.00015 + similarity_loss_val * 10000

            print(
                "#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f Kmeans Loss: %.4f Similarity Loss: %.4f" % (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, kmeans_loss_val / num,
                    similarity_loss_val / num))

            # out = open('test.txt', 'w', encoding='utf8')
            # filename = save_dir

            print_str = "#Epoch %3d: Total: %.4f Clustering Loss: %.4f ZINB Loss: %.4f Kmeans Loss: %.4f Similarity Loss: %.4f" % (
                    epoch + 1, train_loss / num, cluster_loss_val / num, recon_loss_val / num, kmeans_loss_val / num,
                    similarity_loss_val / num)
            # print(result_str)
            result_str = print_str + '\n'
            fh.writelines(result_str)
            fh.flush()

            ml_loss = 0.0

            if epoch % update_ml == 0:
                for ml_batch_idx in range(ml_num_batch):
                    px1 = X[
                        ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw1 = X_raw[ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf1 = sf[
                        ml_ind1[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    px2 = X[
                        ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    sf2 = sf[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    pxraw2 = X_raw[ml_ind2[ml_batch_idx * batch_size: min(ml_num, (ml_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    rawinputs1 = Variable(pxraw1)
                    sfinput1 = Variable(sf1)
                    inputs2 = Variable(px2)
                    rawinputs2 = Variable(pxraw2)
                    sfinput2 = Variable(sf2)
                    z1, q1, mean1, disp1, pi1, _, _ = self.forward(inputs1)
                    z2, q2, mean2, disp2, pi2, _, _ = self.forward(inputs2)
                    loss = (ml_p * self.pairwise_loss(q1, q2, "ML")
                            + self.zinb_loss(rawinputs1, mean1, disp1, pi1, sfinput1)
                            + self.zinb_loss(rawinputs2, mean2, disp2, pi2, sfinput2))
                    ml_loss += loss.data
                    loss.backward()
                    optimizer.step()

            cl_loss = 0.0
            if epoch % update_cl == 0:
                for cl_batch_idx in range(cl_num_batch):
                    px1 = X[cl_ind1[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    px2 = X[cl_ind2[cl_batch_idx * batch_size: min(cl_num, (cl_batch_idx + 1) * batch_size)]]
                    optimizer.zero_grad()
                    inputs1 = Variable(px1)
                    inputs2 = Variable(px2)
                    z1, q1, _, _, _, _, _ = self.forward(inputs1)
                    z2, q2, _, _, _, _, _ = self.forward(inputs2)
                    loss = cl_p * self.pairwise_loss(q1, q2, "CL")
                    cl_loss += loss.data
                    loss.backward()
                    optimizer.step()

            if ml_num_batch > 0 and cl_num_batch > 0:
                print("Pairwise Total:", round(float(ml_loss.cpu()), 2) + float(cl_loss.cpu()), "ML loss",
                      float(ml_loss.cpu()), "CL loss:", float(cl_loss.cpu()))

        return self.y_pred, final_acc, final_nmi, final_ari, final_epoch
