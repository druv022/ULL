from data import TokenizedCorpus, Vocabulary
from embedalign import FFNN, LSTM, ELBO
import torch
from torch.nn import Softplus, Embedding
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal
import time

torch.manual_seed(1)


class Training:

    def __init__(self, data_loc, epochs=30, batch_size = 32, dim_z=32, embedding_dim=128, hidden_dim=100, read=0):
        # data_loc: list of location of L1 and L2;[L1, L2]
        l1, l2 = TokenizedCorpus(data_loc[0]), TokenizedCorpus(data_loc[1])
        self.L1_sentences = l1.get_words("english", read_lines=read)
        self.L2_sentences = l2.get_words("french", read_lines=read)
        self.V1 = Vocabulary(self.L1_sentences,process_all=False)
        self.V2 = Vocabulary(self.L2_sentences,process_all=False)
        self.sentence_length = 5  # max number of tokens in a sentence
        self.L1_data = self.tokenize_data(self.L1_sentences, self.V1.w2i)
        self.L2_data = self.tokenize_data(self.L2_sentences, self.V2.w2i)

        self.epochs = epochs
        self.batch_size = batch_size
        self.dim_Z = dim_z
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        pad = self.V1.w2i["<pad>"]

        # Networks
        self.lstm = LSTM(len(self.V1.w2i), hidden_dim, embedding_dim, pad, bidirectn_flag=True, batch_first=True, batch_size=self.batch_size)
        self.ffnn1 = FFNN(self.dim_Z, int((self.dim_Z + len(self.V1.w2i)) / 6), len(self.V1.w2i))
        self.ffnn2 = FFNN(self.dim_Z, int((self.dim_Z + len(self.V2.w2i)) / 6), len(self.V2.w2i))
        self.ffnn3 = FFNN(hidden_dim, int((hidden_dim + self.dim_Z) / 2), self.dim_Z)
        self.ffnn4 = FFNN(hidden_dim, int((hidden_dim + self.dim_Z) / 2), self.dim_Z)

    def tokenize_sentence(self, sentence, w2i):
        sentence_t = []
        for i, word in enumerate(sentence):
            sentence_t.append(w2i[word])

        len_sentence_t = len(sentence_t)
        if len_sentence_t < self.sentence_length:
            diff = self.sentence_length - len_sentence_t
            pad = [w2i["<pad>"]] * diff
            sentence_t = sentence_t + pad
        else:
            sentence_t = sentence_t[0:self.sentence_length]

        return sentence_t

    def tokenize_data(self, sentences, w2i):
        l_data = []
        for sentence in sentences:
            l_data.append(self.tokenize_sentence(sentence, w2i))

        return torch.Tensor(l_data).long()

    def minibatch(self):

        for i in range(0, len(self.L1_data), self.batch_size):
            yield [self.L1_data[i:i + self.batch_size], self.L2_data[i:i + self.batch_size]]

    def train(self):
        print("-------------Training---------------")
        print("-------------------------------------")
        for epoch in range(self.epochs):
            print("*****************EPOCH ", epoch, "**************************")
            updates = 0
            start = time.time()
            multivariate_n = MultivariateNormal(torch.zeros(self.dim_Z), torch.eye(self.dim_Z))
            for L_batch in self.minibatch():
                updates += 1
                L1_batch = L_batch[0]
                L2_batch = L_batch[1]

                if L1_batch.shape[0] != self.batch_size:
                    continue
                h_1 = self.lstm(L1_batch)

                h = (h_1[:, :, 0:self.hidden_dim] + h_1[:, :, self.hidden_dim:]) / 2

                mu_h = self.ffnn3(h, linear_activation=True)
                self.ffnn4.softmax = Softplus()
                sigma = self.ffnn4(h)

                epsilon = multivariate_n.sample((self.batch_size,self.sentence_length,))
                z = mu_h + epsilon * sigma

                cat_x = self.ffnn1(z)
                cat_y = self.ffnn2(z)

                self.lstm.zero_grad()
                self.ffnn1.zero_grad()
                self.ffnn2.zero_grad()
                self.ffnn3.zero_grad()
                self.ffnn4.zero_grad()

                params = list(self.ffnn1.parameters()) + list(self.ffnn2.parameters()) + list(self.ffnn3.parameters()) \
                         + list(self.ffnn4.parameters()) + list(self.lstm.parameters())
                opt = optim.Adam(params)

                elbo_c = ELBO(self.sentence_length, self.sentence_length)
                elbo_p1 = elbo_c.elbo_p1(cat_x, L1_batch)
                # print(elbo_p1)
                elbo_p2 = elbo_c.elbo_p2(cat_y, L2_batch)
                # print(elbo_p2)
                elbo_p3 = elbo_c.elbo_p3([mu_h, sigma])
                # print(elbo_p3)

                loss = -(elbo_p1 + elbo_p2 - elbo_p3)/self.batch_size

                loss.backward(retain_graph=True)
                opt.step()

                print("iter %r: loss=%.4f, time=%.2fs" %
                      (epoch, loss / updates, time.time() - start))


if __name__ == "__main__":
    # L1_data = "./data/wa/dev.en"
    # L2_data = "./data/wa/dev.fr"

    L1_data = "./data/hansards/training.en"
    L2_data = "./data/hansards/training.fr"

    training = Training([L1_data, L2_data], 1000, batch_size=30, dim_z=100, read=100)
    training.train()
