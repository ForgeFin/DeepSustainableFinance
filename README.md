# DeepSustainableFinance

This repository contains the python code used in the book XY. The according models can be found in the folder "Models". The files containing the file names and the corresponding labels, market capitalization, and industry can be found in the folder "Files". The labels are already split into a training, validation, and test set ("train_data_rs0.txt", "val_data_rs0.txt", and "test_data_rs0.txt"). The enviornmental word list is also contained in the folder "Files".

**Note**: This repository does not contain the 10-K and 10-Q filings. The pre-processed filings can be downloaded from [SRAF](https://sraf.nd.edu/data/stage-one-10-x-parse-data/). The unprocessed files can be downloaded via the [U.S. Securities and Exchange Commission](https://www.sec.gov/Archives/edgar/Feed/).

---


# Requirements
## Installation of required packages

1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Create a virtual environment and install [TensorFlow](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and [PyTorch](https://pytorch.org/get-started/locally/#start-locally).
3. Install [transformers](https://github.com/huggingface/transformers)
4. Install [pandas](https://pandas.pydata.org/docs/getting_started/install.html), [numpy](https://numpy.org/install/), [nltk](https://www.nltk.org/install.html), [scikit-learn](https://scikit-learn.org/stable/install.html), and [lime](https://github.com/marcotcr/lime).

This repository was tested on the following versions: numpy 1.17.0, pandas 1.2.4, tensorflow 2.5.0, keras 2.3.1, nltk 3.4.5, pytorch 1.0.1, and scikit-learn 0.24.2.
Please refer to the according installation pages for the specific install command.
