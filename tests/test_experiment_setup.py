from random import choice
from unittest import TestCase

import pandas as pd
import numpy as np

from helpers.experiment_helper import generate_umap_params, generate_clustering_params
from helpers.experiment_helper import generate_random_experiment


def umap_matches_criteria(umap_parameter):
    is_neighbors_correct = isinstance(umap_parameter.n_neighbors, int)
    is_n_components_correct = isinstance(umap_parameter.n_neighbors, int)
    is_metric_correct = isinstance(umap_parameter.metric, str)

    return is_neighbors_correct and is_n_components_correct and is_metric_correct


def clustering_matches_criteria(clustering_parameter):
    is_size_correct = isinstance(clustering_parameter.min_cluster_size, int)
    is_metric_correct = isinstance(clustering_parameter.metric, str)
    is_selection_method_correct = isinstance(clustering_parameter.cluster_selection_method, str)

    return is_size_correct and is_metric_correct and is_selection_method_correct


class TestExperimentConfiguration(TestCase):

    def test_umap_parameters_quantity(self):
        """
        Only test quantity without verifying the integrity
        """
        n_neighbors_values = range(2, 20)
        n_components_values = range(2, 20)
        metric_values = [
            'euclidean',
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
            'hamming',
            'jaccard',
            'dice',
            'russellrao',
            'kulsinski',
            'rogerstanimoto',
            'sokalmichener',
            'sokalsneath',
            'yule'
        ]

        all_combinations_number = len(metric_values)*len(n_components_values)*len(n_neighbors_values)
        generated_combinations_number = len(generate_umap_params())

        self.assertEqual(generated_combinations_number, all_combinations_number)

    def test_cluster_parameters_quantity(self):
        min_cluster_size_values = range(10, 20)

        metric_values = ['braycurtis',
                         'canberra',
                         'chebyshev',
                         'cityblock',
                         'dice',
                         'euclidean',
                         'hamming',
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

        all_combinations_number = len(min_cluster_size_values) \
                                  * len(metric_values) \
                                  * len(cluster_selection_method_values)
        generated_combinations_number = len(generate_clustering_params())
        self.assertEqual(generated_combinations_number, all_combinations_number)

    def test_umap_parameters_integrity(self):
        """
        Any of the umap paremeters should match the creteria
        """
        umap_params = generate_umap_params()
        meets_criteria = [umap_matches_criteria(umap) for umap in umap_params]
        self.assertTrue(any(meets_criteria))

    def test_clustering_parameters_integrity(self):
        """
        Any of the umap paremeters should match the creteria
        """
        clustering_params = generate_clustering_params()
        meets_criteria = [clustering_matches_criteria(clustering) for clustering in clustering_params]
        self.assertTrue(any(meets_criteria))

    def test_experiment_integrity(self):
        """
        1 experiment = 1 model, 1 set of umap params, 1 set of clustering params
        """
        experiment = generate_random_experiment()

        self.assertTrue(isinstance(experiment.get_result(), pd.DataFrame))

    def test_model_reduction_setup(self):
        experiment = generate_random_experiment()

        reduced_embedding = experiment.reduce_dimensionality()

        assert (reduced_embedding.shape[1]>0)

    def test_model_reduction_integrity(self):
        experiment = generate_random_experiment()

        reduced_embedding = experiment.reduce_dimensionality()

        self.assertEqual(reduced_embedding.shape[1], experiment.n_components)


    def test_clustering_setup(self):
        experiment = generate_random_experiment()
        cluster = experiment.clusterize()

        self.assertIsNotNone(cluster.labels_)

    def test_clustering_integrity(self):
        experiment = generate_random_experiment()
        cluster = experiment.clusterize()
        print (set(cluster.labels_))
        print(len(set(cluster.labels_)))
        self.assertTrue(len(set(cluster.labels_)) > 0)
