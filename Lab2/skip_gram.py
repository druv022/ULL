import numpy as np
from data import TokenizedCorpus, Vocabulary
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
# ----------------------
#Add CUDA 
# ---------------------
CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)
# CUDA = False
# ------------------------

class Training:
    def __init__(self, epochs, training_file, embedding_dim=100,
                                batch_size=32,window_size=2,negative_sample=5):
            self.epochs = epochs
            self.training_file = training_file
            self.embedding_dim = embedding_dim
            self.batch_size = batch_size
            self.win_size = window_size
            self.neg_samples = negative_sample
            corpus = TokenizedCorpus(self.training_file)
            self.sentences = corpus.get_words()
            self.skip_data = Vocabulary(self.sentences, self.win_size)
            self.vocab = self.skip_data.vocab
            self.model = SkipGram(self.vocab, self.embedding_dim)
            self.optimizer = optim.Adam([self.model.in_embed, self.model.out_embed])
            self.training()

    def training(self):
        print("-------------Training---------------")
        print("-------------------------------------")
        for ITER in range(self.epochs):
            updates = 0
            train_loss = 0
            start = time.time()
            for pos_pairs in self.skip_data.minibatch(self.batch_size):
                    updates +=1
                    cen_word = [(idx, pair[0]) for idx, pair in enumerate(pos_pairs)]
                    con_word = [pair[1] for pair in pos_pairs]
                    neg_word = self.skip_data.negative_sampling(pos_pairs)

                    #get one hot representation for center word
                    input_data = self.skip_data.get_onehot(cen_word)
                    #------------------------------------
                    if CUDA:
                        input_data = torch.cuda.FloatTensor(input_data)
                        con_word = torch.cuda.LongTensor(con_word)
                        neg_word = torch.cuda.LongTensor(neg_word)
                    else:
                        input_data  = torch.FloatTensor(input_data)
                        con_word = torch.LongTensor(con_word)
                        neg_word = torch.LongTensor(neg_word)

                    self.loss = self.model.forward(input_data, con_word, neg_word)
                    train_loss+=self.loss[0]
                    self.model.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
            print("iter %r: loss=%.4f, time=%.2fs" %
              (ITER, train_loss/updates, time.time()-start))


class SkipGram(nn.Module):

  """Skip gram model of word2vec.
    Attributes:
        emb_dime: Embedding dimention.
        in_embed: Embedding for center word.
        con_embed: Embedding for context words.
    """

  def __init__(self, vocab, embedding_dim):
      super(SkipGram, self).__init__()
      self.vocab_size = len(vocab)
      self.embedding_dim = embedding_dim
      self.get_tensor()
  
 
  def get_tensor(self):

      if CUDA:
         self.in_embed  = torch.randn(self.embedding_dim, self.vocab_size).cuda().requires_grad_(True)
         self.out_embed = torch.randn(self.vocab_size, self.embedding_dim).cuda().requires_grad_(True)
      else:
         self.in_embed  = torch.randn(self.embedding_dim, self.vocab_size, requires_grad=True)
         self.out_embed = torch.randn(self.vocab_size, self.embedding_dim, requires_grad=True)
         
  

  def forward(self, cen_word, con_word, neg_word):
      #-----------------

      #------------------ 
      
      h_embed = torch.matmul(cen_word, self.in_embed.t())
      h_embed = torch.matmul(h_embed, self.out_embed.t())
      
      #combine neg and con word fro a particular center word
      neg_con = torch.cat((con_word.unsqueeze(-1), neg_word), 1)
      #get score correponding indexes
      h_score = h_embed.gather(1,neg_con)
      #split
      con_score, neg_score = h_score[:,0:1], h_score[:,1:]
      con_score, neg_score = F.logsigmoid(con_score), F.logsigmoid(-neg_score)
      neg_score = torch.sum(neg_score, dim=1).unsqueeze(-1)

      h_score = torch.cat((con_score, neg_score),1) 
      h_score = -1*torch.sum(torch.sum(h_score,dim=1).unsqueeze(-1),dim=0)
      

      return h_score


if __name__ =="__main__":


    train = Training(50, "data/hansards/training.en", 100)

    # # sentences = TokenizedCorpus("data/hansards/training.en")
    # sentences = [['division'], ['poverty', 'also', 'expressed', 'terms',
    #     'increased', 'demand', 'food', 'banks'], ['Bob', 'Speller'], 
    #     ['Mulroney', 'personally', 'demonized', 'patriotism', 'questioned'], ['members'], ['suggest', 'hon', 'chief', 'opposition', 'whip', 'would', 'like', 
    #     'know', 'voted', 'every', 'time', 'House', 
    #     'adjourns', 'Hansard', 'printed', 'list', 'everything']]
        
    # skip_data  = Vocabulary(sentences) 
    # vocab = skip_data.vocab
    # embedding_dim = 5
    # model = SkipGram(vocab, embedding_dim)
    
    # for pos_pairs in skip_data.minibatch(batch_size=10):
        
    #     cen_word = [(idx, pair[0]) for idx, pair in enumerate(pos_pairs)]
    #     con_word = [pair[1] for pair in pos_pairs]
    #     neg_word = skip_data.negative_sampling(pos_pairs)

    #     #get one hot representation for center word
    #     input_data = skip_data.get_onehot(cen_word)
    #     #------------------------------------
    #     if CUDA:
    #         input_data = torch.cuda.FloatTensor(input_data)
    #         con_word = torch.cuda.LongTensor(con_word)
    #         neg_word = torch.cuda.LongTensor(neg_word)
    #     else:
    #         input_data  = torch.FloatTensor(input_data)
    #         con_word = torch.LongTensor(con_word)
    #         neg_word = torch.LongTensor(neg_word)
    #     #----------------------------------------
        
        
        
    #     model.forward(input_data, con_word, neg_word)
        

    #     print()