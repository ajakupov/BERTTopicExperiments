class ClusteringParameters:
    def __init__(self, min_cluster_size, metric, cluster_selection_method):
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

    def to_string(self):
        return "Min Cluster: {}, " \
               "metric {}, " \
               "Cluster Selection: {}, ".format(self.min_cluster_size,
                                   self.metric,
                                   self.cluster_selection_method)