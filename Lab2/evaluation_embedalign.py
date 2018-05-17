import torch
import json
from data import TokenizedCorpus
from torch.distributions.multivariate_normal import MultivariateNormal
from aer import test
import numpy as np
import os

# Tokenize sentence
def tokenize_sentence(sentence, w2i_f,sentence_length = 64):
    sentence_t = []

    for i, word in enumerate(sentence):
        if word in w2i_f:
            sentence_t.append(w2i_f[word])
        # else:
        #     sentence_t.append(w2i_f["<unk>"])

    len_sentence_t = len(sentence_t)
    if len_sentence_t < sentence_length:
        diff = sentence_length - len_sentence_t
        pad = [w2i_f["<pad>"]] * diff
        sentence_t = sentence_t + pad
    else:
        sentence_t = sentence_t[0:sentence_length]

    return sentence_t

# Tokenize data
def tokenize_data(sentences, w2i_f,sentence_length = 64):
    l_data = []
    for sentence in sentences:
        if len(tokenize_sentence(sentence, w2i_f)) > 0:
            l_data.append(tokenize_sentence(sentence, w2i_f,sentence_length=sentence_length))

    return torch.Tensor(l_data).long()


# Read w2i files(filtered as well as unfiltered)
def get_indexes(model_fld):

    with open((os.path.join(model_fld,"L1_w2i.json")),"r") as f:
        L1_w2i = json.load(f)

    with open((os.path.join(model_fld,"L1_i2w.json")),"r") as f:
        L1_i2w = json.load(f)

    with open((os.path.join(model_fld,"L1_w2i_f.json")),"r") as f:
        L1_w2i_f = json.load(f)
        print(len(L1_w2i_f))

    with open((os.path.join(model_fld,"L1_i2w_f.json")),"r") as f:
        L1_i2w_f = json.load(f)

    with open((os.path.join(model_fld,"L2_w2i.json")),"r") as f:
        L2_w2i = json.load(f)

    with open((os.path.join(model_fld,"L2_i2w.json")),"r") as f:
        L2_i2w = json.load(f)

    with open((os.path.join(model_fld,"L2_w2i_f.json")),"r") as f:
        L2_w2i_f = json.load(f)

    with open((os.path.join(model_fld,"L2_i2w_f.json")),"r") as f:
        L2_i2w_f = json.load(f)

    return [L1_w2i, L1_i2w, L1_w2i_f, L1_i2w_f],[L2_w2i, L2_i2w, L2_w2i_f, L2_i2w_f]


# Load models
def load_models(model_fld):
    ffnn1 = torch.load(os.path.join(model_fld,'ffnn1.pt'))
    ffnn2 = torch.load(os.path.join(model_fld,'ffnn2.pt'))
    ffnn3 = torch.load(os.path.join(model_fld,'ffnn3.pt'))
    ffnn4 = torch.load(os.path.join(model_fld,'ffnn4.pt'))

    lstm = torch.load(os.path.join(model_fld,'lstm.pt'))

    return [ffnn1, ffnn2, ffnn3, ffnn4, lstm]

# get minibatch
def get_minibatch(data, batch_size):
    L1_data = data[0]
    L2_data = data[1]
    for i in range(0, len(data), batch_size):
        yield [L1_data[i:i + batch_size], L2_data[i:i + batch_size]]

# Evaluation method for AER Evaluation
def evaluation(models, model_fld,data_loc):
    hidden_dim = 100 # Use the same values as training
    batch_size = 128  # Use the same values as training
    sentence_length = 64 # Could use the same as training
    dim_Z = 150

    ffnn1 = models[0]
    ffnn2 = models[1]
    ffnn3 = models[2]
    ffnn4 = models[3]
    lstm = models[4]

    ffnn1.eval()
    ffnn2.eval()
    ffnn3.eval()
    ffnn4.eval()
    lstm.eval()

    l1, l2 = TokenizedCorpus(data_loc[0]), TokenizedCorpus(data_loc[1])
    L1_sentences = l1.get_words("english")
    L2_sentences = l2.get_words("french")

    V1, V2 = get_indexes(model_fld)

    L1_tokenized = tokenize_data(L1_sentences, V1[2])
    L2_tokenized = tokenize_data(L2_sentences, V2[2])

    pad_l1 = V1[2]["<pad>"]
    pad_l2 = V2[2]["<pad>"]
    batch_counter = 0


    multivariate_n = MultivariateNormal(torch.zeros(dim_Z), torch.eye(dim_Z))
    with open("AER_test.naacl", "w") as aerf:
        for L_batch in get_minibatch([L1_tokenized, L2_tokenized],batch_size):
            L1_batch = L_batch[0]
            L2_batch = L_batch[1]

            # This check is required because the LSTM network depends on fixed batch size
            # if L1_batch.shape[0] != batch_size:
            #     continue

            # Because training was done in batch, the LSTM structure/model seems to be hardcoded for batch
            h_1 = lstm(L1_batch)
            h = (h_1[:, :, 0:hidden_dim] + h_1[:, :, hidden_dim:]) / 2

            epsilon = multivariate_n.sample((batch_size, sentence_length))
            mu = ffnn3(h, linear_activation=True)
            sigma = ffnn4(h)
            z = mu + sigma * epsilon

            for i,l2_sentence in enumerate(L2_batch):
                l1_z = z[i]
                cat_y = ffnn2(l1_z)
                flag = False
                for j, word in enumerate(l2_sentence):
                    if word != pad_l2:
                        flag = True
                        x = cat_y[:,word].unsqueeze(0)
                        x = torch.mul(torch.Tensor(np.where(L1_batch[i,:] > 0, 1, 0)),x)
                        value, index = torch.max(x, 1)
                        aerf.write(str(1+i+batch_counter*batch_size)+" "+ str(int(index)+1) + " " + str(j+1) + " P\n")

        batch_counter += 1


if __name__ == '__main__':

    L1_data = "./data/wa/test.en"
    L2_data = "./data/wa/test.fr"

    # L1_data = "./data/wa/dev.en"
    # L2_data = "./data/wa/dev.fr"

    # model_fld = "model_test"
    model_fld = "model"

    models = load_models(model_fld)
    evaluation(models,model_fld,[L1_data, L2_data])

    gold_path = "./data/wa/test.naacl"
    test_path = "AER_test.naacl"

    test(gold_path,test_path)


