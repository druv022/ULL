# Unsupervised Language Learning

This repository contains code for the [Unsupervised Language Learning](https://uva-slpl.github.io/ull/) course offered 
at the University of Amsterdam  

### Contributors
* [Tarun Krishna](https://github.com/KrishnaTarun)
* [Dhruba Pujary](https://github.com/druv022)

### Lab 1 - Evaluating Word Representations

**Problem Statement:** The goal of this practical is for you to familiarise yourselves with word representation models and different techniques for evaluating them. The word representation model that you will work with is skip-gram, trained using two kinds of context: dependency-based and word window-based. The dependency based model uses dependency annotated context to learn the word representations, as described in the paper by Omer Levy and Yoav Goldberg, [Dependency-based word embeddings, published at ACL 2014](https://aclweb.org/anthology/papers/P/P14/P14-2050/).

* [Report](Lab1/report-ull-lab1.pdf)
* [Code](Lab1/)

### Lab 2 - Learning Word Representations

**Problem Statement:** You will implement 3 models of word representation, one trained for
maximum likelihood, and two latent variable models trained by variational in-
ference. The word representation learning models that you will implement are:
The [Skip-gram](https://arxiv.org/abs/1301.3781), the [Bayesian skip-gram](https://arxiv.org/abs/1711.11027), and [Embed-Align](https://arxiv.org/abs/1802.05883). Skip-gram
is trained discriminatively by having a central word predict context words in a
window surrounding it. Bayesian skip-gram introduces stochastic latent embed-
dings, but does not change the discriminative nature of the training procedure.
Embed-Align introduces stochastic latent embeddings as well as a latent align-
ment variable and learns by generating translation data. Eventually, you should
compare the performance of these three models on the lexical substitution task.

* [Report](Lab2/report-ull-lab2.pdf)
* [Code](Lab2/)

### Lab 3 - Evaluating Sentence Representations

**Problem Statement:** In the 2nd practical, you implemented and trained three different models to
learn the word embeddings: The [Skip-gram](https://arxiv.org/abs/1301.3781), the [Bayesian skip-gram](https://arxiv.org/abs/1711.11027), and [Embed-Align](https://arxiv.org/abs/1802.05883). You have evaluated the performance of these three models on
the lexical substitution task. In this practical, your task is to compare these
models using [SentEval](https://arxiv.org/abs/1803.05449). SentEval, facebook evaluation toolkit for sentence
embeddings, is a library for evaluating the quality of sentence embeddings by
applying them on a broad and diverse set of downstream tasks called ”transfer”
tasks. The reason they are called transfer tasks is that the sentence embeddings
are not explicitly optimized on them.

* [Report](Lab3/report-ull-lab3.pdf)
* [Code](Lab3/)



