import numpy as np
from data import TokenizedCorpus, VsGram
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time
import os
import json
# ----------------------
#Add CUDA 
# ---------------------
CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)
# CUDA = False
# ------------------------

class Training:
    def __init__(self, epochs, training_file, name = "model/skipgram",embedding_dim=100,
                                batch_size=256,window_size=2,negative_sample=5):
            self.epochs = epochs
            self.training_file = training_file
            self.embedding_dim = embedding_dim
            self.batch_size = batch_size
            self.win_size = window_size
            self.neg_samples = negative_sample
            corpus = TokenizedCorpus(self.training_file)
            self.sentences = corpus.get_words()
            self.skip_data = VsGram(self.sentences, self.win_size)
            self.vocab = self.skip_data.w2i
            self.model = SkipGram(self.vocab, self.embedding_dim)
            self.optimizer = optim.SparseAdam(self.model.parameters(), lr=0.001)
            self.name = "model/skipgram"
            
            #save w2i and i2w as json
            with open(os.path.join(self.name+"_i2w.txt"), "w") as out:
                    json.dump(self.skip_data.i2w, out, indent=4)
            with open(os.path.join(self.name+"_w2i.txt"), "w") as out:
                    json.dump(self.skip_data.w2i, out, indent=4)


            self.training()
            

    def training(self):
        print("-------------Training---------------")
        print("-------------------------------------")
        prev_loss = None

        for ITER in range(self.epochs):
            updates=0
            train_loss =0
            start = time.time()
            for pos_pairs in self.skip_data.minibatch(self.batch_size):
                
                    updates+=1
                    # cen_word = [(idx, pair[0]) for idx, pair in enumerate(pos_pairs)]
                    cen_word = [pair[0] for pair in pos_pairs]               
                    con_word = [pair[1:] for pair in pos_pairs]
                    neg_word = self.skip_data.negative_sampling(pos_pairs)

            #         #get one hot representation for center word
            #         input_data = self.skip_data.get_onehot(cen_word)
                    # ------------------------------------
                    if CUDA:
                        # input_data = torch.cuda.FloatTensor(input_data)
                        cen_word = torch.cuda.LongTensor(cen_word)
                        con_word = torch.cuda.LongTensor(con_word)
                        neg_word = torch.cuda.LongTensor(neg_word)
                    else:
                        cen_word = torch.LongTensor(cen_word)
                        # input_data  = torch.FloatTensor(input_data)
                        con_word = torch.LongTensor(con_word)
                        neg_word = torch.LongTensor(neg_word)

                    self.loss = self.model.forward(cen_word, con_word, neg_word)
                    train_loss+=self.loss[0]
                    
                    self.model.zero_grad()
                    self.loss.backward()
                    self.optimizer.step()
            mloss = train_loss/updates
            print("iter %r: loss=%.4f, time=%.2fs" %
              (ITER, mloss, time.time()-start))
            if not prev_loss or mloss < prev_loss:
                prev_loss = mloss
                self.save_embedding(CUDA)
    


    def save_embedding(self, CUDA):
                """Save all embeddings to file.
                
                Args:
                    id2word: map from word id to word.
                    file_name: file name.
                Returns:
                    None.
                """
                file_name = self.name
                if CUDA:
                    embedding = self.model.in_embed.weight.cpu().data.numpy()
                else:
                    embedding = self.model.in_embed.weight.data.numpy()
                
                np.savez(file_name, embedding)
                


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

      self.in_embed =  nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True).cuda() if CUDA \
                        else nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
      self.out_embed = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True).cuda() if CUDA \
                        else nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
      self.init_emb()  
       
  def init_emb(self):
        """ Initialize embedding weight like word2vec.
        Returns:
        None
        """
        initrange = 0.5 / self.embedding_dim
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed.weight.data.uniform_(-0, 0)                 
  
 
         
  

  def forward(self, cen_word, con_word, neg_word):
      #-----------------
     
      #------------------ 
      c_embed = self.in_embed(cen_word)

      p_embed = self.out_embed(con_word)
      n_embed = self.out_embed(neg_word)
      
      p_score = torch.bmm( p_embed, c_embed.unsqueeze(-1)).squeeze(-1)

      p_score = torch.sum(p_score, dim=1).unsqueeze(-1)
      p_score = F.logsigmoid(p_score)
      

      n_score = torch.bmm(n_embed, c_embed.unsqueeze(-1)).squeeze(-1)
      n_score = F.logsigmoid(-1*n_score)
      n_score = torch.sum(n_score, dim=1)

      combine_loss = torch.sum(p_score+n_score, dim=1)
      # loss = comine_loss.sum().unsqueeze(-1)
      

      return -1*torch.mean(combine_loss).unsqueeze(-1)


if __name__ =="__main__":


    train = Training(50, "data/hansards/training.en", 150)
    # train = Training(50, "data/wa/dev.en", 150)

    