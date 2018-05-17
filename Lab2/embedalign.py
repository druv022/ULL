import torch
import torch.nn as nn


torch.manual_seed(1)

# Feed Forward neural network
class FFNN(nn.Module):

    def __init__(self, input_size, output_size,hidden_size=250, hidden_layer = True):
        super(FFNN,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        if hidden_layer:
            self.fc1 = nn.Linear(self.input_size, self.hidden_size)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(self.hidden_size, self.output_size)
        else:
            self.fc1 = nn.Linear(self.input_size, self.output_size)

        self.softmax = nn.Softmax()

    def forward(self, x, linear_activation=False):

        out = self.fc1(x)

        # out = self.relu(out)
        # out = self.fc2(out)

        if not linear_activation:
            out = self.softmax(out)

        return out


# Bi-LSTM network
class LSTM(nn.Module):

    def __init__(self,vocab_size, hidden_dim, embedding_dim, pad, batch_size=1, bidirectn_flag=True, batch_first=False):
        super(LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.batch_first = batch_first

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, pad)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim,batch_first=batch_first, bidirectional=bidirectn_flag)
        self.hidden = self.init_hidden(bidirectn_flag)

    def init_hidden(self, d_flag):
        direction_dim = 1
        if d_flag:
            direction_dim = 2
        return torch.rand([direction_dim, self.batch_size, self.hidden_dim]).requires_grad_(True), torch.rand([direction_dim, self.batch_size, self.hidden_dim]).requires_grad_(True)

    def forward(self, sentence_batch):
        embeds = self.word_embeddings(sentence_batch)

        if len(sentence_batch.shape) > 1:
            seq_length = sentence_batch.shape[1]
        else:
            seq_length = len(sentence_batch)
        if self.batch_first:
            lstm_out, self.hidden = self.lstm(embeds.view(self.batch_size, seq_length, -1), self.hidden)
        else:
            lstm_out, self.hidden = self.lstm(embeds.view(seq_length, 1,-1), self.hidden)

        return lstm_out


# Evalute loss
class ELBO:

    def __init__(self,m,n):
        self.m = m
        self.n = n

    def elbo_p1(self,cat_x, data_l1t, mask):
        # elbo_p1 = np.sum(np.log([cat_x[i, token] for i, token in enumerate(sentence_L1t)]))
        if len(data_l1t.shape) < 2:
            loss = torch.zeros([1])
            for i,token in enumerate(data_l1t):
                loss += torch.log(cat_x[i,token])
        else:
            # data_l1t = torch.mul(data_l1t, mask.long())
            data_l1t = data_l1t.unsqueeze(-1)

            loss = torch.gather(cat_x,2,data_l1t.long())
            loss = torch.log(loss)
            loss = torch.mul(loss,mask.unsqueeze(-1))
            loss = torch.mean(loss)

        # print("Loss 1", loss)
        return loss

    def elbo_p2(self, cat_y, data_l2t, mask):
        # elbo_p2 = np.sum([np.log(np.sum([cat_y[i, sentence_L2t[j]]/m for i in range(0, m)])) for j in range(0, n)])
        loss = torch.zeros([1])
        if len(data_l2t.shape) < 2:
            for i in range(0, self.n):
                loss_temp = torch.zeros([1])
                for j in range(0, self.m):
                    loss_temp += cat_y[j, data_l2t[i]]/self.m

                loss += torch.log(loss_temp)
        else:
            x = data_l2t.unsqueeze(1).repeat([1, self.m, 1])
            loss = torch.gather(cat_y,2,x)
            loss = torch.log(torch.sum(loss, 1)/self.m)
            loss = torch.mul(loss,mask)
            loss = torch.mean(loss)

        # print("Loss 2", loss)
        return loss

    def elbo_p3(self, z_param):
        # elbo_p3 = np.sum([(1 + np.log(z_param[i, 1]**2) - z_param[i, 0]**2 - z_param[i, 1]**2)/2 for i in range(0,m)])
        # kl = torch.zeros([1,1])
        kl_m = torch.zeros([1,1])

        if isinstance(z_param,list):
            mu = z_param[0]
            sigma = z_param[1]
            # print(sigma)
            log_term = torch.log(1/sigma.prod(dim=2))
            # print(log_term)
            trace_term = torch.sum(sigma, 2)
            # print(trace_term)
            mu_term = torch.sum(mu**2, 2)
            # print(mu_term)

            kl_m = 0.5 * (log_term + trace_term + mu_term - mu.size()[-1])

        else:
            for i in range(0, self.m):
                # kl += (1 + torch.log(z_param[i, 1]**2) - z_param[i, 0]**2 - z_param[i, 1]**2)/2
                log_term = torch.log(
                    torch.det(torch.eye(z_param.shape[2])) / torch.det(z_param[i, 1] * torch.eye(z_param.shape[2])))
                trace_term = torch.sum(z_param[i, 1])
                sigma_term = torch.sum(z_param[i, 0] ** 2)
                kl_m = 0.5 * (log_term + trace_term - z_param.shape[2] + sigma_term)
                # kl = 0.5*(np.log(1/np.linalg.det(z_param[i, 1].data.numpy())) - z_param.shape[2] +
                #           np.matrix.trace(np.matmul(np.linalg.inv(z_param[i, 1]),np.eye(z_param.shape[2]))) +
                #           np.matmul(np.matmul((z_param[i,0]).t(), np.linalg.inv(z_param[i, 1])),
                #           z_param[i,0]))

        # print("KL", torch.sum(kl_m))
        return torch.mean(kl_m)


# This method is not tested
class ApproxBiLSTM:

    def __init__(self, vocab_size, embedding_dim, pad,  batch_size=1):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, pad)

    def getEmbedding(self, data):

        embeds = self.word_embeddings(data)
        print("APPROX")
        # print(embeds.shape)
        if len(data.shape) < 2:
            approx = torch.zeros([embeds.shape[0], embeds.shape[1]*2])
            for i, embed in enumerate(embeds):
                if i != 0:
                    other_embeds = torch.cat((embeds[0:i:], embeds[i + 1::]), 0)
                elif i == embeds.shape[0]:
                    other_embeds = embeds[0:-1]
                else:
                    other_embeds = embeds[1:]

                approx[i] = torch.cat((embed, torch.mean(other_embeds, 0)))
        else:

            embeds_n = torch.sum(embeds,dim=1)
            # print(embeds.shape[1])
            embeds_n = embeds_n.unsqueeze(1).repeat([1,embeds.shape[1],1])
            embeds_n = torch.mean(embeds - embeds_n,dim=1)
            embeds_n = embeds_n.unsqueeze(1).repeat([1, embeds.shape[1], 1])

            approx = torch.cat((embeds,embeds_n),dim=2)

        return approx
