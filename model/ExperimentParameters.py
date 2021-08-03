class ExperimentParameters:
    def __init__(self, clustering_parameters, umap_parameters):
        self.min_cluster_size = clustering_parameters.min_cluster_size
        self.clustering_metric = clustering_parameters.metric
        self.cluster_selection_method = clustering_parameters.cluster_selection_method
        self.n_neighbours = umap_parameters.n_neighbours
        self.n_components = umap_parameters.n_components
        self.umap_metric = umap_parameters.metric

    def to_string(self):
        return "Min Cluster: {}, " \
               "metric {}, " \
               "Cluster Selection: {}, " \
               "Neighbours: {}, " \
               "Components: {}, " \
               "Metric: {}".format(self.min_cluster_size,
                                   self.clustering_metric,
                                   self.cluster_selection_method,
                                   self.n_neighbours,
                                   self.n_components,
                                   self.umap_metric)