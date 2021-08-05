import umap
import pandas as pd

class Experiment:
    def __init__(self, sentence_model, umap_parameters, clustering_parameters):
        self.sentence_model = sentence_model
        self.umap_parameters = umap_parameters
        self.clustering_parameters = clustering_parameters
        self.model_reducer = umap.UMAP()

    def get_result(self):
        result = pd.DataFrame()
        result["Model Name"] = pd.Series(self.sentence_model.model_name)

        result["Umap N Neighbours"] = pd.Series(self.umap_parameters.n_neighbours)
        result["Umap N Components"] = pd.Series(self.umap_parameters.n_components)
        result["Umap Metric"] = pd.Series(self.umap_parameters.metric)

        result["Cluster Size"] = pd.Series(self.clustering_parameters.min_cluster_size)
        result["Cluster Metric"] = pd.Series(self.clustering_parameters.metric)
        result["Cluster Selection Method"] = pd.Series(self.clustering_parameters.cluster_selection_method)

        return result

    def reduce_dimensionality(self):
        return None