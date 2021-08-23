class UMAPPArameters:
    def __init__(self, n_neighbors, n_components, metric):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric

    def to_string(self):
        return "Neighbors: {}, " \
               "Components: {}, " \
               "Metric: {}".format(self.n_neighbors,
                                   self.n_components,
                                   self.metric)