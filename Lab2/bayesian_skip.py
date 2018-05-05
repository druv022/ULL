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
# ----------------------
#Add CUDA 
# ---------------------
CUDA = torch.cuda.is_available()
print("CUDA: %s" % CUDA)
CUDA = False
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
            self.sentences = corpus.get_words()[:5]
            self.skip_data = VbsGram(self.sentences, self.win_size)
            print(len(self.skip_data.data))
            self.vocab = self.skip_data.w2i
            print(len(self.vocab))
            self.model = BaysianGram(self.vocab, self.embedding_dim)
            #TODO optimizer
            
            # params = [self.model.infer_, self.model.infer_mu.parameters()\
            #           ,self.model.infer_sig.parameters()]
            # print(params)
            # for params in self.model.parameters():
            #     print(params)

            self.optimizer = optim.Adam(self.model.parameters())
            self.training()

    def training(self):
        print("-------------Training---------------")
        print("-------------------------------------")
        for ITER in range(1):

            updates = 0
            train_loss = 0
            start = time.time()
            
            for pos_pairs in self.skip_data.minibatch(self.batch_size):
                print("******")
                # print(pos_pairs)
                
                updates +=1
                cen_word = [(idx, pair[0]) for idx, pair in enumerate(pos_pairs)]
                
                con_word = [ (idx, c_id, w ) for idx, pair in enumerate(pos_pairs) \
                                         for c_id, w in enumerate(pair[1:])]
                # print()
                id_con =  np.array([pair[1:] for  pair in pos_pairs])   
                # print(id_con)
                mask = np.where(id_con>0,1,0)
                # #get one hot representation for center word
                in_cen = self.skip_data.get_onehot(cen_word, len(cen_word))
                in_con = self.skip_data.get_onehot(con_word, len(cen_word),self.win_size*2)
                #------------------------------------
                #TODO send context id for mapping
                if CUDA:
                    in_cen = torch.cuda.FloatTensor(in_cen)
                    in_con = torch.cuda.FloatTensor(in_con)
                    id_con = torch.cuda.LongTensor(id_con)
                    mask = torch.cuda.FloatTensor(mask)
                    
                else:
                    in_cen  = torch.FloatTensor(in_cen)
                    in_con = torch.FloatTensor(in_con)
                    id_con = torch.LongTensor(id_con)
                    mask = torch.FloatTensor(mask)
                con_dist, mu_post, sig_post, pr_mu, pr_sig=self.model.forward(in_cen, in_con, \
                                                           self.win_size)
                self.loss = self.model.loss(con_dist, mu_post, \
                      sig_post, pr_mu, pr_sig, id_con, mask)
            
                train_loss+=self.loss[0]
                #backprop    
                self.model.zero_grad()
                self.loss.backward()
                self.optimizer.step()
            
            print("iter %r: loss=%.4f, time=%.2fs" %
                  (ITER, train_loss/updates, time.time()-start))


class BaysianGram(nn.Module):

  """Skip gram model of word2vec.
    Attributes:
        emb_dime: Embedding dimention.
        in_embed: Embedding for center word.
        con_embed: Embedding for context words.
    """

  def __init__(self, vocab, embedding_dim):
      super(BaysianGram, self).__init__()
      self.vocab_size = len(vocab)

      self.embed_dim = embedding_dim
      self.z_embed  = embedding_dim #this dimension could be differnt

      #----------encoder : Inference all intialiizations belong to encoder----------
      infer_ = torch.randn(self.embed_dim, 
                self.vocab_size).cuda() if CUDA else torch.randn(self.embed_dim, 
                self.vocab_size)
      self.infer_ = nn.Parameter(infer_, requires_grad=True)
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
      prior_mu = torch.randn(self.z_embed, 
                self.vocab_size).cuda() if CUDA else \
                torch.randn(self.embed_dim, 
                self.vocab_size)
      self.prior_mu = nn.Parameter(prior_mu, requires_grad=True)

      prior_sig = torch.randn(self.z_embed, 
                self.vocab_size).cuda() if CUDA else \
                torch.randn(self.embed_dim, 
                self.vocab_size)
      self.prior_sig = nn.Parameter(prior_sig, requires_grad=True)          
      #---------------------------

  

  def forward(self, cen_word, con_word, window_size):
      #-----------------
      span = window_size*2
      #------------------ 
      #--------Encoder-----------------------
      print("Infer_t",self.infer_.t())
      h_cen = torch.matmul(cen_word, self.infer_.t())
      h_con = torch.matmul(con_word, self.infer_.t())
      
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
      sig_post =F.softplus(self.infer_sig(h))
      sig_post = sig_post**2
      #-----------------------------------------
      
      #-----------Decoder------------------
      
      #Parametrization trick
      
      z_embed = mu_post + torch.mul(self.eps.sample((sig_post.size()[0],)), sig_post)
      con_dist = F.softmax(self.generat(z_embed), dim=1)
      # print(con_dist)
      # con_dist.clamp(max=1e-4)
      #---------------------------------------
      
      #---------------Prior Network---------------
      pr_mu  = torch.matmul(cen_word, self.prior_mu.t())
      pr_sig = torch.matmul(cen_word,self.prior_sig.t())
      pr_sig = F.softplus(pr_sig)**2
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

        c_dis = torch.gather(context_distribution,1 , id_)
        log_expec = torch.log(c_dis)
        #perfrom masking
        log_expec = torch.sum(log_expec*mask,dim=1).unsqueeze(-1)
        # print("LOG loss ter::::::",log_expec)
        #Get KL loss 
        #--------------------------------------
        #log of ratio of determinant of sigma
        log_term = torch.log(prior_sig.prod(dim=1).unsqueeze(-1)/\
                             post_sig.prod(dim=1).unsqueeze(-1))
        inv_ = 1./prior_sig
        #multiplication of two diaagonal matrix is a diagonal matrix
        trace_term =  torch.sum(inv_*post_sig, dim=1).unsqueeze(-1)
        sigma_term = torch.sum((prior_mu - post_sig)**2/inv_, dim=1).unsqueeze(-1)
        kl_loss = 0.5*(log_term -prior_sig.size()[-1]\
                       + trace_term+sigma_term)

        # print("KL loss Term:::::::",kl_loss)
        loss = torch.sum(log_expec-kl_loss, dim=0)
        # print(loss)
        return loss
        

        # #--------------------------------------
        # t1 = (1/2)*torch.log(prior_sig/post_sig)
        # t2_num = ((post_mu-prior_mu)**2) + post_sig**2
        # t2_den = 2*torch.sqrt(prior_sig) 
        # t2 = t2_num/t2_den
        # kl_loss = torch.sum(t2+t1-0.5,dim=1).unsqueeze(-1)
        




      

      # return h_score


if __name__ =="__main__":


    train = Training(50, "data/wa/dev.en", 70)

   