import numpy as np
from data import TokenizedCorpus, VbsGram
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as distb
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
    def __init__(self, epochs, training_file, embedding_dim=100,
                                batch_size=128,name = "model/bsg",window_size=2):
            self.epochs = epochs
            self.training_file = training_file
            self.embedding_dim = embedding_dim
            self.batch_size = batch_size
            self.win_size = window_size
            corpus = TokenizedCorpus(self.training_file)
            self.sentences = corpus.get_words()[:]
            self.skip_data = VbsGram(self.sentences, self.win_size)
            self.vocab = self.skip_data.w2i
            PAD= self.skip_data.PAD
            self.model = BayesianGram(self.vocab, self.embedding_dim, PAD)
            
            self.name = name
            
            #save w2i and i2w as json
            with open(os.path.join(self.name+"_i2w.txt"), "w") as out:
                    json.dump(self.skip_data.i2w, out, indent=4)
            with open(os.path.join(self.name+"_w2i.txt"), "w") as out:
                    json.dump(self.skip_data.w2i, out, indent=4)

            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.training()

    
    
    def training(self):
        


        
        print("---------------------------------------------")
        print("\t\tTraining")
        print("---------------------------------------------")
        prev_loss = None
        for ITER in range(self.epochs):

            updates = 0
            train_loss = 0
            start = time.time()
            
            for pos_pairs in self.skip_data.minibatch(self.batch_size):
                
                # print(pos_pairs)
                
                updates +=1
                # cen_word = [(idx, pair[0]) for idx, pair in enumerate(pos_pairs)]
                cen_word = [pair[0] for  pair in pos_pairs]
                
                # con_word = [ (idx, c_id, w ) for idx, pair in enumerate(pos_pairs) \
                #                          for c_id, w in enumerate(pair[1:])]
                con_word = [ pair[1:] for pair in pos_pairs]
                # print()
                id_con =  np.array([pair[1:] for  pair in pos_pairs])   
                
                mask = np.where(id_con>0,1,0)
                # # #get one hot representation for center word
                # in_cen = self.skip_data.get_onehot(cen_word, len(cen_word))
                # in_con = self.skip_data.get_onehot(con_word, len(cen_word),self.win_size*2)
                #------------------------------------
                #TODO send context id for mapping
                if CUDA:
                    # in_cen = torch.cuda.FloatTensor(in_cen)
                    # in_con = torch.cuda.FloatTensor(in_con)
                    cen_word = torch.cuda.LongTensor(cen_word)
                    con_word = torch.cuda.LongTensor(con_word)
                    id_con = torch.cuda.LongTensor(id_con)
                    mask = torch.cuda.FloatTensor(mask)
                    
                else:
                    # in_cen  = torch.FloatTensor(in_cen)
                    # in_con = torch.FloatTensor(in_con)
                    cen_word = torch.LongTensor(cen_word)
                    con_word = torch.LongTensor(con_word)
                    id_con = torch.LongTensor(id_con)
                    mask = torch.FloatTensor(mask)
                con_dist, mu_post, sig_post, pr_mu, pr_sig=self.model.forward(cen_word, con_word, \
                                                           self.win_size)
                self.loss = self.model.loss(con_dist, mu_post, \
                      sig_post, pr_mu, pr_sig, id_con, mask)
                
                train_loss+=self.loss[0]
                
                #backprop    
                self.model.zero_grad()
                self.loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e-2)
                self.optimizer.step()
            
            mloss = train_loss/updates
            print("iter %r: loss=%.4f, time=%.2fs" %
              (ITER, mloss, time.time()-start))
            if not prev_loss or mloss < prev_loss:
                prev_loss = mloss
                with open(self.name, 'wb') as f:
                      torch.save(self.model, f)
                # self.save_embedding(CUDA)


class BayesianGram(nn.Module):

  """Skip gram model of word2vec.
    Attributes:
        emb_dime: Embedding dimention.
        in_embed: Embedding for center word.
        con_embed: Embedding for context words.
    """

  def __init__(self, vocab, embedding_dim, PAD):
      super(BayesianGram, self).__init__()
      self.vocab_size = len(vocab)

      self.embed_dim = embedding_dim
      self.z_embed  = embedding_dim #this dimension could be differnt

      #----------encoder : Inference all intialiizations belong to encoder----------
      # infer_ = torch.randn(self.embed_dim, 
      #           self.vocab_size).cuda() if CUDA else torch.randn(self.embed_dim, 
      #           self.vocab_size)
      # self.infer_ = nn.Parameter(infer_, requires_grad=True)

      self.infer_ =  nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=PAD).cuda() if CUDA \
                        else nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=PAD)
      
      self.infer_mu = nn.Linear(2*self.embed_dim, self.z_embed, bias=True).cuda() if CUDA \
                      else nn.Linear(2*self.embed_dim, self.z_embed, bias=True)
      self.infer_sig = nn.Linear(2*self.embed_dim, self.z_embed, bias=True).cuda() if CUDA \
                        else nn.Linear(2*self.embed_dim, self.z_embed, bias=True)
      #---------------------------
      # create an epsilon N(0,I)
      sig = torch.diag(torch.ones(self.z_embed)).cuda() if CUDA \
            else torch.diag(torch.ones(self.z_embed))
      mu  = torch.zeros(self.z_embed).cuda() if CUDA \
            else torch.zeros(self.z_embed) 
      self.eps = distb.MultivariateNormal(mu, sig)
      #----------------------------
      # #--------decoder------------
      self.generat = nn.Linear(self.z_embed, self.vocab_size, bias=True).cuda() if CUDA \
                    else  nn.Linear(self.z_embed, self.vocab_size, bias=True)
      # #---------------------------

      # #-------prior---------------
      # prior_mu = torch.randn(self.z_embed, 
      #           self.vocab_size).cuda() if CUDA else \
      #           torch.randn(self.embed_dim, 
      #           self.vocab_size)
      # self.prior_mu = nn.Parameter(prior_mu, requires_grad=True)
      self.prior_mu =  nn.Embedding(self.vocab_size, self.z_embed, padding_idx=PAD).cuda() if CUDA \
                        else nn.Embedding(self.vocab_size, self.z_embed, padding_idx=PAD)
      
      # prior_sig = torch.randn(self.z_embed, 
      #           self.vocab_size).cuda() if CUDA else \
      #           torch.randn(self.embed_dim, 
      #           self.vocab_size)
      self.prior_sig = nn.Embedding(self.vocab_size, self.z_embed, padding_idx=PAD).cuda() if CUDA \
                        else nn.Embedding(self.vocab_size, self.z_embed, padding_idx=PAD)          
      #---------------------------
      self.init_emb()

  def init_emb(self):
      """ Initialize embedding weight like word2vec.
      Returns:
      None
      """
      initrange = 0.5 / self.embed_dim
      self.infer_.weight.data.uniform_(-initrange, initrange)
      
      initrange =   0.5 / self.z_embed
      self.prior_mu.weight.data.uniform_(-initrange, initrange) 
      self.prior_sig.weight.data.uniform_(-initrange, initrange)
  
  
          
  
  def forward(self, cen_word, con_word, window_size):
      #-----------------
      span = window_size*2
      
      #------------------ 
      #--------Encoder-----------------------
      # print(cen_word)
      
      # self.get_new_weights()
      # self.infer_.weight.data.clamp_(min=-1e-3)
      h_cen = self.infer_(cen_word)
      h_con = self.infer_(con_word)
      # print(h_cen)
      # print(h_cen)
      #concat center word and with it's context word
      #-----------------------------------
      
      h_cen = h_cen.unsqueeze(-1).transpose(1,2)
      h_cen = h_cen.repeat(1,span, 1)
      
      #concat
      h = torch.cat((h_cen, h_con),2)

      #--------------------------------
      h = F.relu(h)
      h = torch.sum(h, dim=1)
      #calulate mu and sigma for q(z|x,c) for approximating p(z|x)
      mu_post = self.infer_mu(h)
      # sig_post =F.softplus(self.infer_sig(h))
      # sig_post = torch.exp(self.infer_sig(h))
      sig_post = self.infer_sig(h)
      # print(sig_post)
      # sig_post = sig_post
      # print("**post*** \n",mu_post)
      # print(sig_post)
      # print()

      #-----------------------------------------
      
      #-----------Decoder------------------
      
      #Parametrization trick
      
      z_embed = mu_post + self.eps.sample((sig_post.size()[0],))* torch.sqrt(torch.exp(sig_post))
      con_dist = F.log_softmax(self.generat(z_embed), dim=1)
      # print(con_dist)
      # print(con_dist)
      # con_dist.clamp(max=1e-3)
      #---------------------------------------
      
      #---------------Prior Network---------------
      pr_mu  = self.prior_mu(cen_word)
      pr_sig = self.prior_sig(cen_word)
      pr_sig = F.softplus(pr_sig)
     
      #-----------------------------------------------

      return con_dist, mu_post, sig_post, pr_mu, pr_sig

  @staticmethod
  def loss(context_distribution, post_mu, post_sig, \
                prior_mu, prior_sig, id_, mask):

        """
          Context_distribution = batch_size x V
          post_mu = batch_size x Z_dim
          post_sig = batch_size x Z_dim
          prior_mu = batch_size x Z_dim
          prior_sigma =batch_size x Z_dim
        """
        #for the loss we have expectation term and KL term
        #Expectation

        log_expec = torch.gather(context_distribution,1 , id_)
        # log_expec = torch.log(c_dis)
        #perfrom masking
        log_expec = torch.sum(log_expec*mask,dim=1).unsqueeze(-1)
        # print("LOG loss ter::::::",log_expec)
        #Get KL loss 
        #--------------------------------------
        t1 = torch.log(prior_sig) - post_sig*(1/2)
        t2_num = ((post_mu-prior_mu)**2) + torch.exp(post_sig)
        t2_den = 2*(prior_sig**2) 
        t2 = ((t2_num/t2_den)-0.5)
        kl_loss = torch.sum(t2+t1,dim=1).unsqueeze(-1)
        # #---------------------------------  -----
        # print("KL:::losss",kl_loss)
        loss = log_expec-kl_loss
        
        return -1*torch.mean(loss).unsqueeze(-1)
        

        



        # log_term = torch.log(prior_sig.prod(dim=1).unsqueeze(-1)/\
        #                      post_sig.prod(dim=1).unsqueeze(-1))
        # inv_ = 1./prior_sig
        # #multiplication of two diagonal matrix is a diagonal matrix
        # trace_term =  torch.sum(inv_*post_sig, dim=1).unsqueeze(-1)
        # sigma_term = torch.sum((prior_mu - post_sig)**2/inv_, dim=1).unsqueeze(-1)

        # kl_loss = 0.5*(log_term -prior_sig.size()[-1]\
        #                + trace_term+sigma_term)

      

      # return h_score


if __name__ =="__main__":

    train = Training(50, "data/hansards/training.en", 150)

    # train = Training(50, "data/wa/dev.en", 300)

   