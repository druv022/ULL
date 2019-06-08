# Unsupervised Language Learning

This repository contains code for the [Unsupervised Language Learning](https://uva-slpl.github.io/ull/) course offered 
at the University of Amsterdam  

### Contributors
* [Tarun Krishna](https://github.com/KrishnaTarun)
* [Dhruba Pujary](https://github.com/druv022)

### Lab 1 - Evaluating Word Representations

**Problem Statement:** The goal of this practical is for you to familiarise yourselves with word representation models and different techniques for evaluating them. The word representation model that you will work with is skip-gram, trained using two kinds of context: dependency-based and word window-based. The dependency based model uses dependency annotated context to learn the word representations, as described in the paper by Omer Levy and Yoav Goldberg, [Dependency-based word embeddings, published at ACL 2014](https://aclweb.org/anthology/papers/P/P14/P14-2050/).

* [Report](lab1/report-ull-lab1.pdf)
* [Code](lab1/)

### Lab 2 - Learning Word Representations

**Problem Statement:** You will implement 3 models of word representation, one trained for
maximum likelihood, and two latent variable models trained by variational in-
ference. The word representation learning models that you will implement are:
The [skip-gram](https://arxiv.org/abs/1301.3781), the [Bayesian skip-gram](https://arxiv.org/abs/1711.11027), and [Embed-Align](https://arxiv.org/abs/1802.05883). Skip-gram
is trained discriminatively by having a central word predict context words in a
window surrounding it. Bayesian skip-gram introduces stochastic latent embed-
dings, but does not change the discriminative nature of the training procedure.
Embed-Align introduces stochastic latent embeddings as well as a latent align-
ment variable and learns by generating translation data. Eventually, you should
compare the performance of these three models on the lexical substitution task.

* [Report](lab2/report-ull-lab2.pdf)
* [Code](lab2/)

