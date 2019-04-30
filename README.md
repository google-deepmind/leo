# Meta-Learning with Latent Embedding Optimization

## Overview
This repository contains the implementation of the meta-learning model
described in the paper "[Meta-Learning with Latent Embedding
Optimization](https://arxiv.org/abs/1807.05960)" by Rusu et. al. It was posted
on arXiv in July 2018 and will be presented at ICLR 2019.

The paper learns a data-dependent latent representation of model parameters and
performs gradient-based meta-learning in this low-dimensional space.

The code here doesn't include the (standard) method for pre-training the
data embeddings. Instead, the trained embeddings are provided.

Disclaimer: This is not an official Google product.

## Running the code

### Setup
To run the code, you first need to need to install:

- [TensorFlow](https://www.tensorflow.org/install/) and [TensorFlow Probability](https://www.tensorflow.org/probability) (we used version 1.12),
- [Sonnet](https://github.com/deepmind/sonnet) (we used version v1.29), and
- [Abseil](https://github.com/abseil/abseil-py) (we use only the FLAGS module).

### Getting the data
You need to download [the embeddings](http://storage.googleapis.com/leo-embeddings/embeddings.zip) and extract them on disk:

```
$ wget http://storage.googleapis.com/leo-embeddings/embeddings.zip
$ unzip embeddings.zip
$ EMBEDDINGS=`pwd`/embeddings
```

### Running the code
Then, clone this repository using:

`$ git clone https://github.com/deepmind/leo`

and run the code as:

`$ python runner.py --data_path=$EMBEDDINGS`

This will train the model for solving 5-way 1-shot miniImageNet classification.

### Hyperparameters
To train the model on the tieredImageNet dataset or with a different number of
training examples per class (K-shot), you can pass these parameters with
command-line or in `config.py`, e.g.:

`$ python runner.py -- --data_path=$EMBEDDINGS --dataset_name=tieredImageNet
--num_tr_examples_per_class=5 --outer_lr=1e-4`

See `config.py` for the list of options to set.

Comparison of paper and open-source implementations in terms of test set accuracy:

| Implementation         | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
| -----------------------| ------------------- | ------------------- | --------------------- | --------------------- |
| `LEO Paper`            |   `61.76 ± 0.08%`   |   `77.59 ± 0.12%`   |    `66.33 ± 0.05%`    |    `81.44 ± 0.09%`    |
| `This code`            |   `61.89 ± 0.16%`   |   `77.65 ± 0.09%`   |    `66.25 ± 0.14%`    |    `81.77 ± 0.09%`    |


The hyperparameters we found working best for different setups are as follows:

| Hyperparameter                 | miniImageNet 1-shot | miniImageNet 5-shot | tieredImageNet 1-shot | tieredImageNet 5-shot |
| ------------------------------ | ------------------- | ------------------- | --------------------- | --------------------- |
| `outer_lr`                     |    `2.739071e-4`    |    `4.102361e-4`    |     `8.659053e-4`     |     `6.110314e-4`     |
| `l2_penalty_weight`            |    `3.623413e-10`   |    `8.540338e-9`    |     `4.148858e-10`    |     `1.690399e-10`    |
| `orthogonality_penalty_weight` |      `0.188103`     |    `1.523998e-3`    |     `5.451078e-3`     |     `2.481216e-2`     |
| `dropout_rate`                 |      `0.307651`     |     `0.300299`      |      `0.475126`       |      `0.415158`       |
| `kl_weight`                    |      `0.756143`     |     `0.466387`      |     `2.034189e-3`     |      `1.622811`       |
| `encoder_penalty_weight`       |    `5.756821e-6`    |    `2.661608e-7`    |     `8.302962e-5`     |     `2.672450e-5`     |
