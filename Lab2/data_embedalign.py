from data import TokenizedCorpus, Vocabulary
from embedalign import FFNN, LSTM, ELBO
import torch
from torch.nn import  Softplus, Embedding
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

torch.manual_seed(1)

def tokenize_sentence(sentence, w2i):
    sentence_t = []
    for i, word in enumerate(sentence):
        sentence_t.append(w2i[word])
    return sentence_t

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    print('output size:', output.data.size())
    print('output norm:', output.data.norm())
    print('-*-*-*-*-*-*-------------------------------')

def printgradnorm(self, grad_input, grad_output):
    print('Inside ' + self.__class__.__name__ + ' backward')
    print('Inside class:' + self.__class__.__name__)
    print('')
    print('grad_input: ', type(grad_input))
    print('grad_input[0]: ', type(grad_input[0]))
    print('grad_output: ', type(grad_output))
    print('grad_output[0]: ', type(grad_output[0]))
    print('')
    print('grad_input size:', grad_input[0].size())
    print('grad_output size:', grad_output[0].size())
    print('grad_input norm:', grad_input[0].norm())
    print('------------------------------------------')


L1_data = "./data/wa/dev.en"
L2_data = "./data/wa/dev.fr"

dim_Z = 2

# sentences
L1 = TokenizedCorpus(L1_data)
L2 = TokenizedCorpus(L2_data)

L1_sentences = L1.get_words("english")
L2_sentences = L2.get_words("french")

# Test code
L1_sentences = [L1_sentences[0]]
L2_sentences = [L2_sentences[0]]

# vocabulary, w2i, i2w, unigram
V1 = Vocabulary(L1_sentences)
V2 = Vocabulary(L2_sentences)

print(V1.vocab)
print(V2.vocab)

z_table = dict()

for epoch in range(30):
    for x, sentence_L1 in enumerate(L1_sentences):
        sentence_L1t = torch.Tensor(tokenize_sentence(sentence_L1, V1.w2i)).long().requires_grad_(True)
        # print(sentence_L1t)
        sentence_L2 = L2_sentences[x]
        sentence_L2t = torch.Tensor(tokenize_sentence(sentence_L2, V2.w2i)).long().requires_grad_(True)
        # print(sentence_L2t)

        # L1 sentence length
        print(sentence_L1t)
        m = len(sentence_L1t)
        # L2 sentence length
        print(sentence_L2t)
        n = len(sentence_L2t)

        multivariate_n = MultivariateNormal(torch.zeros(dim_Z), torch.eye(dim_Z))

        # Inference Network --------------------------------------------------------------------------------------------
        hidden_dim = 4
        embedding_dim = 5

        # bow = BOW(len(V1.w2i), embedding_dim)
        # h_11 = bow(sentence_L1t)
        # print(h_11)
        # h = [h_11]

        # print("@#@@#",len(V1.w2i), hidden_dim, embedding_dim)
        lstm_1 = LSTM(len(V1.w2i), hidden_dim, embedding_dim)
        h_1 = lstm_1(sentence_L1t)

        lstm_1.register_forward_hook(printnorm)
        lstm_1.register_backward_hook(printgradnorm)

        z_param = torch.zeros([m,2,dim_Z])

        # print("@@@@",h_1.shape,h_1,"\n",h_1[:,:,0:hidden_dim],h_1[:,:,hidden_dim:])
        h = (h_1[:,:,0:hidden_dim] + h_1[:,:,hidden_dim:])/2
        # print("*#*", len(h),h[0].squeeze())
        print("chain 1 ", h.requires_grad)

        print("ffnn3-4", len(h[0].squeeze()), int((len(h[0].squeeze()) + dim_Z) / 2), dim_Z)

        ffnn3 = FFNN(len(h[0].squeeze()), int((len(h[0].squeeze()) + dim_Z)/2), dim_Z)
        ffnn4 = FFNN(len(h[0].squeeze()), int((len(h[0].squeeze()) + dim_Z)/2), dim_Z)

        ffnn3.register_forward_hook(printnorm)
        ffnn4.register_forward_hook(printnorm)
        ffnn3.register_backward_hook(printgradnorm)
        ffnn4.register_backward_hook(printgradnorm)

        print(len(h))

        for i in range(0, len(h)):
            # print("***",h[i].squeeze())
            print("FOR ffnn3 i",i)

            mu_h = ffnn3(h[i].squeeze(), linear_activation = True)
            # print("Chain 2 ", mu_h.requires_grad)

            print("FOR ffnn4 i", i)
            ffnn4.softmax = Softplus()
            var_h = ffnn4(h[i].squeeze())
            # print("Chain 3 ", var_h.requires_grad)

            epsilon = multivariate_n.sample()
            z = mu_h + epsilon * var_h
            z_table[i] = z
            # print(z_param.shape, z_param,)
            z_param[i,0,:], z_param[i,1,:] = mu_h, var_h
            # print("Chain 5", z_param[i,0,:].requires_grad)
            # print("Chain 6", z_param[i, 1, :].requires_grad)

        # Generative network -------------------------------------------------------------------------------------------
        cat_x = torch.zeros(m, len(V1.w2i))
        cat_y = torch.zeros(m, len(V2.w2i))

        print("fnn1", dim_Z,int((dim_Z + len(V1.w2i))/2),len(V1.w2i))
        print("fnn2", dim_Z, int((dim_Z + len(V2.w2i)) / 2), len(V2.w2i))

        ffnn1 = FFNN(dim_Z, int((dim_Z + len(V1.w2i))/2), len(V1.w2i))
        ffnn2 = FFNN(dim_Z, int((dim_Z + len(V2.w2i))/2), len(V2.w2i))

        ffnn1.register_forward_hook(printnorm)
        ffnn2.register_forward_hook(printnorm)
        ffnn1.register_backward_hook(printgradnorm)
        ffnn2.register_backward_hook(printgradnorm)

        print(m)

        for i in range(0, m):
            if i not in z_table.keys():
                z = multivariate_n.sample()
            else:
                z = z_table[i]

            # get categorical distribution
            print("FOR ffnn1 i",i )
            cat_x[i, :] = ffnn1(z)
            print("FOR ffnn2 i", i)
            cat_y[i, :] = ffnn2(z)
            # print("Chain 7", cat_x[i, :].requires_grad)
            # print("Chain 8", cat_y[i, :].requires_grad)

        # ------------------------------------------------
        ffnn1.zero_grad()
        ffnn2.zero_grad()
        ffnn3.zero_grad()
        ffnn4.zero_grad()

        print(list(ffnn1.parameters()), "\n\n2: ", list(ffnn2.parameters()), "\n\n3: ", list(ffnn3.parameters()),
        "\n\n4: ", list(ffnn4.parameters()), "\n\nL: ", list(lstm_1.parameters()))
        params = list(ffnn1.parameters()) + list(ffnn2.parameters()) + list(ffnn3.parameters()) + list(ffnn4.parameters())#\
                 #+ list(lstm_1.parameters())
        opt = optim.Adam(params)
        #

        # LOSS function ------------------------------------------------------------------------------------------------
        elbo_c = ELBO(m,n)
        elbo_p1 = elbo_c.elbo_p1(cat_x, sentence_L1t)
        # print("Chain 9 ",elbo_p1.requires_grad)
        elbo_p2 = elbo_c.elbo_p2(cat_y, sentence_L2t)
        # print("Chain 10 ", elbo_p2.requires_grad)
        elbo_p3 = elbo_c.elbo_p3(z_param)
        # print("Chain 11 ", elbo_p3.requires_grad)
        # print(elbo_p1.shape, elbo_p2.shape, elbo_p3.shape)
        loss = -(elbo_p1 + elbo_p2 - elbo_p3)
        print("#*#*", loss)

        loss.backward()
        opt.step()
