from data import TokenizedCorpus, Vocabulary
from embedalign import FFNN, LSTM, ELBO
import torch
from torch.nn import NLLLoss, Softplus
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

torch.manual_seed(1)

def tokenize_sentence(sentence, w2i):
    sentence_t = []
    for i, word in enumerate(sentence):
        sentence_t.append(w2i[word])
    return sentence_t


L1_data = "./data/wa/dev.en"
L2_data = "./data/wa/dev.fr"

dim_Z = 1

# sentences
L1 = TokenizedCorpus(L1_data)
L2 = TokenizedCorpus(L2_data)

L1_sentences = L1.get_words("english")
L2_sentences = L2.get_words("french")

# vocabulary, w2i, i2w, unigram
V1 = Vocabulary(L1_sentences)
V2 = Vocabulary(L2_sentences)

z_table = dict()

for epoch in range(30):
    for x, sentence_L1 in enumerate(L1_sentences):
        sentence_L1t = torch.Tensor(tokenize_sentence(sentence_L1, V1.w2i)).long().requires_grad_(True)
        # print(sentence_L1t)
        sentence_L2 = L2_sentences[x]
        sentence_L2t = torch.Tensor(tokenize_sentence(sentence_L2, V2.w2i)).long().requires_grad_(True)
        # print(sentence_L2t)

        # L1 sentence length
        m = len(sentence_L1t)
        # L2 sentence length
        n = len(sentence_L2t)

        multivariate_n = MultivariateNormal(torch.zeros(dim_Z), torch.eye(dim_Z))

        # Inference Network --------------------------------------------------------------------------------------------
        hidden_dim = 4
        embedding_dim = 5

        lstm_1 = LSTM(len(V1.w2i), hidden_dim, embedding_dim)
        h_1 = lstm_1(sentence_L1t)
        # lstm_2 = LSTM(len(V1.w2i), hidden_dim, embedding_dim)
        # inv_idx = torch.arange(sentence_L1t.size(0)-1, -1, -1).long()
        # inv_tensor = sentence_L1t.index_select(0, inv_idx)
        # h_2 = lstm_2(inv_tensor)
        #
        # print("***",h_1,type(h_1))
        # print("***",h_2)
        # h = h_1 + h_2
        z_param = torch.zeros([m,2,dim_Z])

        h = (h_1[:,:,0:7] + h_1[:,:,7:])/2
        # print("*#*", len(h),h[0].squeeze())
        print("chain 1 ", h.requires_grad)

        ffnn3 = FFNN(len(h[0].squeeze()), int((len(h[0].squeeze()) + dim_Z)/2), dim_Z)
        # print(len(h), int((len(h) + dim_Z)/2), dim_Z)
        ffnn4 = FFNN(len(h[0].squeeze()), int((len(h[0].squeeze()) + dim_Z)/2), dim_Z)

        for i in range(0, len(h)):
            # print("***",h[i].squeeze())
            mu_h = ffnn3(h[i].squeeze())
            print("Chain 2 ", mu_h.requires_grad)

            ffnn4.softmax = Softplus()
            var_h = ffnn4(h[i].squeeze())
            print("Chain 3 ", var_h.requires_grad)

            epsilon = multivariate_n.sample()
            z = mu_h + epsilon * var_h
            z_table[i] = z
            # print(z_param.shape, z_param,)
            z_param[i,0,:], z_param[i,1,:] = mu_h, var_h
            print("Chain 5", z_param[i,0,:].requires_grad)
            print("Chain 6", z_param[i, 1, :].requires_grad)

        # ffnn3.zero_grad()
        # elbo_c = ELBO(m, n)
        # opt3 = optim.Adam(ffnn3.parameters(), lr=0.01)
        # elbo_p3 = elbo_c.elbo_p3(z_param)
        # print(elbo_p3)
        # elbo_p3.backward()
        # opt3.step()

        # Generative network -------------------------------------------------------------------------------------------
        cat_x = torch.zeros(m, len(V1.w2i))
        cat_y = torch.zeros(m, len(V2.w2i))

        ffnn1 = FFNN(dim_Z, int((dim_Z + len(V1.w2i))/2), len(V1.w2i))
        ffnn2 = FFNN(dim_Z, int((dim_Z + len(V2.w2i))/2), len(V2.w2i))

        for i in range(0, m):
            if i not in z_table.keys():
                z = multivariate_n.sample()
            else:
                z = z_table[i]

            # get categorical distribution
            cat_x[i, :] = ffnn1(z)
            cat_y[i, :] = ffnn2(z)
            print("Chain 7", cat_x[i, :].requires_grad)
            print("Chain 8", cat_y[i, :].requires_grad)

        # ------------------------------------------------
        ffnn1.zero_grad()
        opt1 = optim.Adam(ffnn1.parameters(), lr=0.01)
        ffnn2.zero_grad()
        opt2 = optim.Adam(ffnn2.parameters(), lr=0.01)
        # ffnn3.zero_grad()
        # opt3 = optim.Adam(ffnn3.parameters(), lr=0.01)
        # ffnn4.zero_grad()
        # opt4 = optim.Adam(ffnn4.parameters(), lr=0.01)
        #
        # # LOSS function ------------------------------------------------------------------------------------------------
        elbo_c = ELBO(m,n)
        elbo_p1 = elbo_c.elbo_p1(cat_x, sentence_L1t)
        # print("Chain 9 ",elbo_p1.requires_grad)
        elbo_p2 = elbo_c.elbo_p2(cat_y, sentence_L2t)
        # print("Chain 10 ", elbo_p2.requires_grad)
        # elbo_p3 = elbo_c.elbo_p3(z_param)
        # print("Chain 11 ", elbo_p3.requires_grad)
        # print(elbo_p1.shape, elbo_p2.shape, elbo_p3.shape)
        loss = -(elbo_p1 + elbo_p2 )#- elbo_p3)
        # print("#*#*",loss)
        #
        loss.backward()
        opt1.step()
        opt2.step()
        # opt3.step()
        # opt4.step()
