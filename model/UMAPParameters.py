class UMAPPArameters:
    def __init__(self, n_neighbours, n_components, metric):
        self.n_neighbours = n_neighbours
        self.n_components = n_components
        self.metric = metric

    def to_string(self):
        return "Neighbours: {}, " \
               "Components: {}, " \
               "Metric: {}".format(self.n_neighbours,
                                   self.n_components,
                                   self.metric)