# Exploring SQuAD with different model architectures
The [Stanford Question Answer Dataset](https://rajpurkar.github.io/SQuAD-explorer/) (SQuAD) is a recently released reading comprehension dataset. In this repository, I aim to reimplement state of the art network architectures that have been successful on the SQuAD dataset. The current rankings can be found [here](https://rajpurkar.github.io/SQuAD-explorer/).

[//]: # (Image References)
[image1]: ./README-files/Baseline-model/loss.png
[image2]: ./README-files/Baseline-model/EM_scores.png
[image3]: ./README-files/Baseline-model/f1_scores.png
[image4]: ./README-files/Attention-model/loss.png
[image5]: ./README-files/Attention-model/EM_scores.png
[image6]: ./README-files/Attention-model/f1_scores.png
[image7]: ./README-files/BiDAF-model/loss.png
[image8]: ./README-files/BiDAF-model/EM_scores.png
[image9]: ./README-files/BiDAF-model/f1_scores.png



## Models implemented
I have currently implemented 3 models:
* Standard encoder-decoder using BiLSTMs for both the encoder and decoder to be used for the baseline
* Encoder-decoder using the [attention mechanism](https://arxiv.org/pdf/1508.04025.pdf)
* [Bi-Directional Attention Flow](https://arxiv.org/pdf/1611.01603.pdf) (BiDAF) (not fully complete, still have to add CNNs for the character level embeddings)

## Prerequisites 
* Tensorflow v1.1 (preferably with GPU if you want to do your own training)

Note that although it isn't essential it is highly recommended to use some package manager such as [Anaconda](https://anaconda.org/) so that different Python environments can be kept in parallel without interfering with each other.

## Getting started 
1. Download this repo 
2. Download the dataset from [here](https://drive.google.com/open?id=0B77UOMTOybVZeWtnTF94QWJjekk) and drag the `data` folder to the root of the project repo
3. To train a model simply `cd` to the root folder of the repo project and type: 

> python train.py --model Baseline --train_dir path/to/save/results


Note here that the only two required flags are the `--model` flag and the `--train_dir` flag, the other flags have default values and can be found in the `train.py` file.

## Results

In all F1 and EM graphs, red are the results from the validation dataset, whereas blue are the results from the training sets.

### Baseline model
![alt text][image1]
![alt text][image2]
![alt text][image3]

### Luong's Attention model
![alt text][image4]
![alt text][image5]
![alt text][image6]

### Bi-directional Attention flow model
![alt text][image7]
![alt text][image8]
![alt text][image9]