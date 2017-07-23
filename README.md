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
* Standard Seq2Seq using BiLSTMs for both the encoder and decoder to be used for the baseline
* Seq2Seq with added [attention mechanism](https://arxiv.org/pdf/1508.04025.pdf)
* [Bi-Directional Attention Flow](https://arxiv.org/pdf/1611.01603.pdf) (BiDAF) (not fully complete, still have to add CNNs for the character level embeddings)

## Prerequisites 
* [Anaconda](https://anaconda.org/) to install the packages
* Tensorflow v1.1 (preferably with GPU if you want to do your own training) I had to work with Tensorflow v1.1 as in v1.2 support for GPUs were disabled for Macs.
> pip install 'tensorflow-gpu==1.1.0'

## Installation
1. Create and setup conda environment
> source create -n squad python=3
2. Activate the created environment
> source activate squad
3. Install Tensorflow
> pip install 'tensorflow-gpu==1.1.0'
4. Install Matplotlib
> conda install matplotlib
5. Install Jupyter notebook (if you want to run the Notebook to see a demo of the trained neural network)
> conda install notebook ipykernel
6. Create the kernel to be used by the notebook
> ipython kernel install --user --display-name squad --name squad

## Getting started (for training)
1. Download this repo 
2. Download the dataset from [here](https://drive.google.com/open?id=0B77UOMTOybVZeWtnTF94QWJjekk) and drag the `data` folder to the root of the project repo
3. To train a model simply `cd` to the root folder of the repo project and type: 

> python train.py --model Baseline --train_dir path/to/save/results

Note here that the only two required flags are the `--model` flag and the `--train_dir` flag, the other flags have default values and can be found in the `train.py` file.

Another important flag is the `--eval_num` flag which specifies how often the model is saved

## Getting started (for notebook)
1. Go to the root folder and type

> jupyter notebook

Then click on `LuongAttention.ipynb` and run through the notebook. It will load up the model, choose a validation sample from the dataset and predict on it which will then be compared to the correct answer.


## Results

In all F1 and EM graphs, red are the results from the validation dataset, whereas blue are the results from the training sets.

### Baseline model

Batch size used: 128

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Luong's Attention model

Batch size used: 256

The Attention module achieves ~71% for F1 and ~58% for EM for the validation set. This is comparable to the Match-LSTM model by Singapore Management University 
![alt text][image4]
![alt text][image5]
![alt text][image6]

### Bi-directional Attention flow model

Batch size used: 24

The BiDAF module achieves similar scores to the Attention module, although a better score should be able to be achieved after the character level CNN embedding layer is implemented.

![alt text][image7]
![alt text][image8]
![alt text][image9]
