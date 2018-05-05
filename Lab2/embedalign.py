import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import numpy as np

torch.manual_seed(1)


class FFNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(FFNN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax()

    def forward(self, x, linear_activation=False):

        # print("*#",x.shape)

        out = self.fc1(x)

        # print("*#", out.shape)

        out = self.relu(out)
        out = self.fc2(out)
        # print(out.shape)
        if not linear_activation:
            out = self.softmax(out)

        return out


class LSTM(nn.Module):

    def __init__(self,vocab_size, hidden_dim, embedding_dim, bidirectn_flag=True):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectn_flag)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.rand([2, 1, self.hidden_dim]).requires_grad_(True), torch.rand([2, 1, self.hidden_dim]).requires_grad_(True)

    def forward(self, sentence):
        # print("@#@#@", sentence)
        embeds = self.word_embeddings(sentence)
        # print("@#@#@", embeds)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        print("@#@#@#@",self.hidden)

        # seq_len, batch, hidden_dim * directions
        return lstm_out



class ELBO:

    def __init__(self,m,n):
        self.m = m
        self.n = n

    def elbo_p1(self,cat_x, sentence_l1t):
        # elbo_p1 = np.sum(np.log([cat_x[i, token] for i, token in enumerate(sentence_L1t)]))
        loss = torch.zeros([1])
        for i,token in enumerate(sentence_l1t):
            loss += torch.log(cat_x[i,token])

        print("Loss 1", loss)
        return loss

    def elbo_p2(self, cat_y, sentence_l2t):
        # elbo_p2 = np.sum([np.log(np.sum([cat_y[i, sentence_L2t[j]]/m for i in range(0, m)])) for j in range(0, n)])
        loss = torch.zeros([1])
        for i in range(0, self.n):
            loss_temp = torch.zeros([1])
            for j in range(0, self.m):
                loss_temp += cat_y[j, sentence_l2t[i]]/self.m

            loss += torch.log(loss_temp)

        print("Loss 2", loss)
        return loss

    def elbo_p3(self, z_param):
        # elbo_p3 = np.sum([(1 + np.log(z_param[i, 1]**2) - z_param[i, 0]**2 - z_param[i, 1]**2)/2 for i in range(0,m)])
        kl = torch.zeros([1,1])
        kl_m = torch.zeros([1,1])
        for i in range(0, self.m):
            # print(i,1 + torch.log(z_param[i, 1]**2),torch.log(z_param[i, 1]**2) ,z_param[i, 0],z_param[i, 0]**2,"***", z_param[i, 1], z_param[i, 1]**2)
            # kl += (1 + torch.log(z_param[i, 1]**2) - z_param[i, 0]**2 - z_param[i, 1]**2)/2
            # print(kl)

            # print(z_param[i, 1]*torch.eye(z_param.shape[2]))
            log_term = torch.log(torch.det(torch.eye(z_param.shape[2]))/torch.det(z_param[i, 1]*torch.eye(z_param.shape[2])))
            # print(log_term)
            trace_term = torch.sum(z_param[i, 1])
            # print(trace_term)
            sigma_term = torch.sum(z_param[i, 0]**2)
            kl_m = 0.5 * (log_term + trace_term - z_param.shape[2] + sigma_term)
            # kl = 0.5*(np.log(1/np.linalg.det(z_param[i, 1].data.numpy())) - z_param.shape[2] +
            #           np.matrix.trace(np.matmul(np.linalg.inv(z_param[i, 1]),np.eye(z_param.shape[2]))) +
            #           np.matmul(np.matmul((z_param[i,0]).t(), np.linalg.inv(z_param[i, 1])),
            #           z_param[i,0]))

        print("KL ", kl_m)
        return kl_m
