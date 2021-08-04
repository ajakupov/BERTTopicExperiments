# Topic Modeling Experiments

Run full experimentation pipeline of BERT-based topic modeling with all the parameters and experimentation logging. 
The approach is inspired by Maarten Grootendorst's 
[article](https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6), 
which, in turn, has been inspired by the [Top2Vec](https://github.com/ddangelov/Top2Vec) project. The goal of his 
research was to propose a topic modeling algorithm based on BERT and hugging-face transformer embeddings. In his 
tutorial he already provided the configuration for UMAP embedding, clustering, and, which is more important, 
Sentence BERT model. However, for the research, it may be sometimes interesting to create a pipeline and construct a 
grid of hyperparameters that are fine-tuned during the experimentation. Thus, I've decided to create this public 
repository, and steadily, step-by-step, implement this pipeline, to be able to make a profound 
models/hyperparameters comparison. 


## Install packages and requirements
*(As certain packages are only compatible with Linux environment, all the commands mentioned below will be 
Linux-based. Windows version is, unfortunately, not that stable).*


Before installing the python packages, install ```python-dev``` tools.
```commandline
sudo apt-get install python3.8-dev
```
This will avoid the following installation error
>ERROR: Could not build wheels for hdbscan which use PEP 517 and cannot be installed directly

To run the solution on your environment:
1. Install Python virtual environment
```
pip install virtualenv
```
2. Go to the project folder (where you have your README.md file)
3. Create your virtual environment
```
python3 -m venv bertvent
```
4. Access to venv virtual environment
```
source bertvent/bin/activate
```
5. Install project dependencies
```
pip install -r requirements.txt
```
6. Install other packages and update your requirements with the following command
```
pip freeze>requirements.txt
```