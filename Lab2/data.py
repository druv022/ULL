import numpy as np
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import torch


np.random.seed(7)

class TokenizedCorpus:
    """Reads input files, removes punctuation, stopwiords and returns a sentence iterator """

    def __init__(self, corpus):
        self.filename = corpus
        

    def get_words(self):
        porter = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        self.input_file = open(self.filename) 

        sentences = []
        
        for line in self.input_file:
            
             tokens = word_tokenize(line)
             
             # Filterout Punctuation and Stopwords
             line = [word for word in tokens if word.isalpha() if word not in stop_words]
             sentences.append(line)
        print("---Sentences Processed-------")
        print()
        return sentences
             
class VsGram:

    def __init__(self, sentences, window_size=2):
        #Type for Skip_gram:SG and Bayesian skip gram: BSG

        # create mapping from word-2-index and index-2-word
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict(lambda: len(self.i2w))
        self.vocab =[]
        # store frequency of words
        self.unigram = {}
        self.build_vocab(sentences)
        self.get_context(sentences, window_size) 
        self.negative_sampling_table()
        UNK = self.w2i["<unk>"]
        PAD = self.w2i["<pad>"]
        self.w2i = defaultdict(lambda: UNK, self.w2i)

    def build_vocab(self, sentences):
        for sen in sentences:
            for token in sen:
                try:
                    self.unigram[token]+=1
                except:
                    self.unigram[token]=1


                self.w2i[token]
                self.i2w[self.w2i[token]] = token
                # create vocubulary i.e uniuqe words
                if token not in self.vocab:
                    self.vocab.append(token)
        
        # print(len(self.unigram.keys()), len(self.vocab))
    
    
    def get_context(self, sentences, window_size =2):
        # This is our data which is to be fed into NN i.e word_context_pair
        self.data = []
        span = window_size*2+1
        for sen in sentences:
            # print(sen)
            indx = [self.w2i[token] for token in sen]
            # print(indx)
            for idx, cen_word in enumerate(indx):
                for w in range(-window_size, window_size+1):
                   
                    context_pos = w+ idx
                    if context_pos<0 or context_pos==idx or context_pos>=len(indx):
                        continue
                    self.data.append((cen_word, indx[context_pos]))
                
                  

           
    def get_onehot(self,word_idx):
        x = np.zeros((len(word_idx),len(self.vocab)))
        for l in word_idx:
            x[l] = 1.0
        return x
    # TODO batch_size

    def minibatch(self, batch_size=32):
        
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i+batch_size]


    #TODO : subsamplig  
    #TODO: still look into this 
    def negative_sampling_table(self):
        
        self.table = []
        self.table_size = sum(Counter(self.unigram).values())
        
        # heuristics for negative sampling
        U_fre = np.array(list(self.unigram.values()))**(0.75)
        Z = sum(U_fre)
        P = U_fre/Z

        #get distribution of words in the table
        contri = np.round(P*self.table_size)
        
        for idx, c in enumerate(contri):
            self.table+=[idx]*np.int(c)
    
    
    def negative_sampling(self, pairs, negative_samples=5):
        neg_word = np.random.choice(self.table, size = (len(pairs),negative_samples))
        return neg_word 

class VbsGram:
    def __init__(self, sentences, window_size=2):
        

        # create mapping from word-2-index and index-2-word
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = defaultdict(lambda: len(self.i2w))
        self.PAD = self.w2i["<pad>"]
        self.UNK = self.w2i["<unk>"]
        self.build_vocab(sentences)
        self.get_context(sentences, window_size) 
        self.w2i = defaultdict(lambda: UNK, self.w2i)

    def build_vocab(self, sentences):
        for sen in sentences:
            for token in sen:
                self.w2i[token]
                self.i2w[self.w2i[token]] = token
                        
          
    
    def get_context(self, sentences, window_size =2):
        # This is our data which is to be fed into NN i.e word_context_pair
        self.data = []
        span = window_size*2+1
        for sen in sentences:
            # print(sen)
            indx = [self.w2i[token] for token in sen]
            # print(indx)
            for idx, cen_word in enumerate(indx):
                
                word_context = [cen_word] 
                # print(cen_word, self.i2w[cen_word])
                for w in range(-window_size, window_size+1):
                   
                    context_pos = w+ idx
                    # print("Context_pos = ",context_pos)
                                            
                    if context_pos<0 or context_pos==idx or context_pos>=len(indx):
                       continue
                    else:
                       word_context.append(indx[context_pos]) 
                # if len(word_context)< span:
                #                                                    
                if len(word_context)>1 :
                    word_context+=[self.PAD]*(span-len(word_context))
                    self.data.append(word_context)
                
                       

           # print(self.data)
            # print()
    def get_onehot(self,word_idx, batch, multi=None):
        #incoming batch may be equal to batch_size
        if multi==None:
            x = np.zeros((batch,len(self.w2i)))
        else:
            x = np.zeros((batch, multi, len(self.w2i)))
        for l in word_idx:
            x[l] = 1.0
        return x
    # TODO batch_size

    def minibatch(self, batch_size=32):
        
        for i in range(0, len(self.data), batch_size):
            yield self.data[i:i+batch_size]


    
if __name__ == "__main__":
    # a = TokenizedCorpus("data/hansards/training.en")
    # sentences = a.get_words()
    # print(sentences[0:10])
    sentences = [['division'], ['poverty', 'also', 'expressed', 'terms',
        'increased', 'demand', 'food', 'banks'], ['Bob', 'Speller'], 
        ['Mulroney', 'personally', 'demonized', 'patriotism', 'questioned'], ['members'], ['suggest', 'hon', 'chief', 'opposition', 'whip', 'would', 'like', 'know', 'voted', 'every', 'time', 'House', 'adjourns', 'Hansard', 'printed', 'list', 'everything'],
        ['urge', 'environment', 'minister', 'least', 'visit',
        'site', 'talk', 'concerned', 'citizens', 'appreciate', 'firsthand', 'important']]

    d = VocabularyBayesianSkipGram(sentences)
    for i in d.minibatch():
        print(i)
        print()