"""
    SSVAE Trainer
"""
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import kl_divergence, log_mean_exp, save_model, save_vars
from regularisers import MMD_DIM

class SSVAETrainer:
    """
        SSVAE Trainer Class
    """
    def __init__(self, model, load_model=False,
                 filepath="./models/",
                 modelname=None,
                 train_dataset=None,
                 test_dataset=None,
                 regs=MMD_DIM(), K=1,
                 beta=1., alpha=1.,
                 lr=1e-4, amsgrad=True, beta1=0.9, beta2=0.999,
                 cuda_n="cuda:0", today=datetime.datetime.now().date()):
        """ Initialization """
        self.model = model  # StyleScore VAE
        self.load_model = load_model
        self.filepath = filepath

        self.trainset = None
        self.testset = None

        self.get_datasets(train_dataset, test_dataset)

        self.K = K
        self.beta = beta
        self.alpha = alpha

        self.cuda_n = cuda_n
        self.today = today
        self.regs = regs
        self.pred = nn.MSELoss

        # optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                                 self.model.parameters()),
                                                 lr=lr, amsgrad=amsgrad,
                                                 betas=(beta1, beta2))
        self.lr = lr
        self.amsgrad = amsgrad
        self.beta1 = beta1
        self.beta2 = beta2

        if torch.cuda.is_available() and cuda_n[:4] == "cuda":
            self.device = cuda_n
        else:
            self.device = "cpu"

        self.model = self.model.to(self.device)

        if load_model:
            self.load_our_model(filepath + modelname)

        self.train_results = defaultdict(list)

    def train(self, epochs=20, batch_size=64):
        self.model.train()

        # Set random train index
        train_idx = np.arange(self.trainset.shape[0])
        np.random.shuffle(train_idx)

        train_N = train_idx.shape[0]

        length = train_idx.shape[0] // batch_size

        for epoc in range(epochs):
            len_t = tqdm(range(length))
            obj_loss = 0.
            recon_loss = 0.
            overlap_loss = 0.
            sparsity_loss = 0.

            for j in len_t:
                train_i = train_idx[j * batch_size:(j + 1) * batch_size]
                trainset = self.trainset[train_i]

                trainset = trainset.to(self.device)

                self.optimizer.zero_grad()
                obj, recon, overlap, sparsity = \
                        self.times_objective(trainset, K=self.K,
                                             beta=self.beta, alpha=self.alpha)
                obj = -obj
                obj.backward()
                self.optimizer.step()
                obj_loss += obj.item()
                recon_loss += recon.item()
                overlap_loss += overlap.item()
                sparsity_loss += sparsity.item()

                des = "{} obj_loss: {:.2f}, recon: {:.2f}, overlap: {:.4f}, "
                des += "sparsity: {:.4f}"
                len_t.set_description(des.format(epoc,
                                                 obj.item() / batch_size,
                                                 recon.item() / batch_size,
                                                 overlap.item() / batch_size,
                                                 sparsity.item() / batch_size,
                                                 ))

            self.train_results['train_obj_loss'].append(obj_loss / train_N)
            self.train_results['train_recon_loss'].append(recon_loss / train_N)
            self.train_results['train_overlap_loss'].\
                    append(overlap_loss / train_N)
            self.train_results['train_sparsity_loss'].\
                    append(sparsity_loss / train_N)

            self.save_our_model()
        save_vars(self.train_results, self.model_file_name[:-3] + "_losses.rar")

    def get_datasets(self, trainset, testset):
        " get datasets "

        trainset = torch.FloatTensor(trainset.astype(float))
        testset = torch.FloatTensor(testset.astype(float))
        self.trainset = trainset
        self.testset = testset

    def load_our_model(self, file_name):
        " load model "
        self.model.load_state_dict(torch.load(file_name))
        self.model = self.model.to(self.device)
        self.model.eval()

    def save_our_model(self, file_name=None):
        " save model "
        if file_name is not None:
            save_model(self.model, file_name)
        else:
            today = self.today.strftime("%Y%m%d")
            file_name = self.filepath + self.model.modelName + \
                    "_" + "beta_" + str(self.beta) + "_alpha_" + \
                    str(self.alpha) + "_" + today + ".pt"
            save_model(self.model, file_name)
        self.model_file_name = file_name

    def times_objective(self, mf, K=1, beta=1., alpha=1.):
        """
            Optimizer

            Args:
                model: our model
                mf: Multifactors
                rets: realized returns
        """
        qz_x, px_z, style_score = self.model(mf, K=K)

        # Reconstruction
        lpx_z = px_z.log_prob(mf).sum(-1)
        pz = self.model.pz(*self.model.pz_params)

        # Overlap
        kld = kl_divergence(qz_x, pz, samples=style_score).sum(-1)

        # Sparsity
        reg = (self.regs(pz.sample(torch.Size([mf.size(0)])).\
                view(-1, style_score.size(-1)), style_score) \
                         if self.regs.samples else self.regs(pz, qz_x))

        obj = lpx_z - (beta * kld) - (alpha * reg)

        recon = lpx_z.sum()
        overlap = kld.sum()
        sparsity = reg.sum()

        return obj.sum(), recon, overlap, sparsity
