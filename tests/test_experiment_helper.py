import logging
from unittest import TestCase

from helpers.experiment_helper import generate_umap_params, generate_clustering_params
from model.UMAPParameters import UMAPPArameters


class TestExperimentConfiguration(TestCase):

    def test_umap_parameters_quantity(self):
        """
        Only test quantity without verifying the integrity
        """
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

        all_combinations_number = len(metric_values)*len(n_components_values)*len(n_neighbours_values)
        generated_combinations_number = len(generate_umap_params())

        self.assertEqual(generated_combinations_number, all_combinations_number)

    def test_cluster_parameters_quantity(self):
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
        meets_criteria = [self.umap_matches_criteria(umap) for umap in umap_params]
        self.assertTrue(any(meets_criteria))

    def umap_matches_criteria(self, umap_parameter):
        is_neighbours_correct = isinstance(umap_parameter.n_neighbours, int)
        is_n_components_correct = isinstance(umap_parameter.n_neighbours, int)
        is_metric_correct = isinstance(umap_parameter.metric, str)

        return is_neighbours_correct and is_n_components_correct and is_metric_correct