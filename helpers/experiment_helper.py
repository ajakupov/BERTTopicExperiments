import itertools

from random import choice
from sentence_transformers import SentenceTransformer

from model.ClusteringParameters import ClusteringParameters
from model.UMAPParameters import UMAPPArameters
from model.SentenceModel import SentenceModel


def generate_umap_params():
    n_neighbours_values = range(2, 100)
    n_components_values = range(2, 100)
    metric_values = ['euclidean',
                     'manhattan',
                     'chebyshev',
                     'minkowski',
                     'canberra',
                     'braycurtis',
                     'mahalanobis',
                     'wminkowski',
                     'seuclidean',
                     'cosine',
                     'correlation',
                     'haversine',
                     'hamming',
                     'jaccard',
                     'dice',
                     'russelrao',
                     'kulsinski',
                     'll_dirichlet',
                     'hellinger',
                     'rogerstanimoto',
                     'sokalmichener',
                     'sokalsneath',
                     'yule']

    combinations_list = [n_neighbours_values, n_components_values, metric_values]

    parameters_list = []

    for element in itertools.product(*combinations_list):
        (n_neighbours, n_components, metric) = element
        umap_paremeter = UMAPPArameters(n_neighbours, n_components, metric)
        parameters_list.append(umap_paremeter)

    return parameters_list


def generate_clustering_params():
    min_cluster_size_values = range(1, 20)

    metric_values = ['braycurtis',
                     'canberra',
                     'chebyshev',
                     'cityblock',
                     'dice',
                     'euclidean',
                     'hamming',
                     'haversine',
                     'infinity',
                     'jaccard',
                     'kulsinski',
                     'l1',
                     'l2',
                     'mahalanobis',
                     'manhattan',
                     'matching',
                     'minkowski',
                     'p',
                     'pyfunc',
                     'rogerstanimoto',
                     'russellrao',
                     'seuclidean',
                     'sokalmichener',
                     'sokalsneath',
                     'wminkowski']

    cluster_selection_method_values = ['eom', 'leaf']

    combinations_list = [min_cluster_size_values, cluster_selection_method_values, metric_values]

    parameters_list = []

    for element in itertools.product(*combinations_list):
        (min_cluster_size, cluster_selection_method, metric) = element
        cluster_parameter = ClusteringParameters(min_cluster_size, metric, cluster_selection_method)
        parameters_list.append(cluster_parameter)

    return parameters_list


def construct_sentence_model(model_name, data):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(data, show_progress_bar=True)

    sentence_model = SentenceModel(model_name, embeddings)
    return sentence_model


def generate_models(data):
    pre_trained_models = ['paraphrase-mpnet-base-v2',
                          'paraphrase-multilingual-mpnet-base-v2',
                          'paraphrase-TinyBERT-L6-v2',
                          'paraphrase-distilroberta-base-v2',
                          'paraphrase-MiniLM-L12-v2',
                          'paraphrase-MiniLM-L6-v2',
                          'paraphrase-albert-small-v2',
                          'paraphrase-multilingual-MiniLM-L12-v2',
                          'paraphrase-MiniLM-L3-v2',
                          'nli-mpnet-base-v2',
                          'stsb-mpnet-base-v2',
                          'distiluse-base-multilingual-cased-v1',
                          'stsb-distilroberta-base-v2',
                          'nli-roberta-base-v2',
                          'stsb-roberta-base-v2',
                          'nli-distilroberta-base-v2',
                          'distiluse-base-multilingual-cased-v2',
                          'average_word_embeddings_komninos',
                          'average_word_embeddings_glove.6B.300d']

    sentence_models = []

    for model in pre_trained_models:
        sentence_models.append(construct_sentence_model(model, data))

    return sentence_models


def generate_random_model(data):
    pre_trained_models = ['paraphrase-mpnet-base-v2',
                          'paraphrase-multilingual-mpnet-base-v2',
                          'paraphrase-TinyBERT-L6-v2',
                          'paraphrase-distilroberta-base-v2',
                          'paraphrase-MiniLM-L12-v2',
                          'paraphrase-MiniLM-L6-v2',
                          'paraphrase-albert-small-v2',
                          'paraphrase-multilingual-MiniLM-L12-v2',
                          'paraphrase-MiniLM-L3-v2',
                          'nli-mpnet-base-v2',
                          'stsb-mpnet-base-v2',
                          'distiluse-base-multilingual-cased-v1',
                          'stsb-distilroberta-base-v2',
                          'nli-roberta-base-v2',
                          'stsb-roberta-base-v2',
                          'nli-distilroberta-base-v2',
                          'distiluse-base-multilingual-cased-v2',
                          'average_word_embeddings_komninos',
                          'average_word_embeddings_glove.6B.300d']

    random_model_name = choice(pre_trained_models)
    random_sentence_embedding = construct_sentence_model(random_model_name, data)

    random_sentence_model = SentenceModel(random_model_name, random_sentence_embedding)

    return random_sentence_model
