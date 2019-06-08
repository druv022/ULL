## Lab 3 - Evaluating Sentence Representations

  This folder contains four .ipynb file. __Skipgram_SentEval.ipynb__ file is used for analyzing sentence embedding for Skip-gram, __senteval_embedalign.ipynb__ is the one provided along with EmbedAlign word embeddings. 
 
  Skipgram model is trained using gensim. To generate Skipgram embeddings copy the data to data folder, change the data path 
  in  w2vec_gensim.ipynb file and run the file. This will generate the embedding.

**Steps to follow**:

  For **Skip-gram analysis** :
  1. Copy the trained Skip-gram model into the models folder.
  2. Select the type of transfer task. The trasfer task are categorized based on downstream and probing task.
  3. Run the .ipynb file. Results will be printed as well as a pickle dump of the results gets generated.
  
  For **EmbedAligned**:
  1. Instruction provided in the assignment documents. Copy the file in the EmbedAligned notebook folder.
  2. Select the same set of task as Skipgram.
  3. Run the .ipynb file. Results will be printed as well as a pickle dump of the results gets generated.
  
  
Skipgram_SentEval_different_param.ipynb is used to analyze Skipgram models with different word embedding size.


**Note**: The Process folder contains a support scripts to analyze the pickle dumps at later stage. This scripts can be modified by user according to his/her need. This script is used to obtain plots for analysis.

**Author** : Dhruba Pujary, Tarun Krishna
  
