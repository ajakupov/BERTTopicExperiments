import umap
import pandas as pd


class Experiment:
    def __init__(self, sentence_model, umap_parameters, clustering_parameters):
        print(umap_parameters.to_string())
        self.sentence_model = sentence_model
        self.n_neighbors = umap_parameters.n_neighbors
        self.n_components = umap_parameters.n_components
        self.metric = umap_parameters.metric
        self.clustering_parameters = clustering_parameters

    def get_result(self):
        result = pd.DataFrame()
        result["Model Name"] = pd.Series(self.sentence_model.model_name)

        result["Umap N Neighbors"] = pd.Series(self.n_neighbors)
        result["Umap N Components"] = pd.Series(self.n_components)
        result["Umap Metric"] = pd.Series(self.metric)

        result["Cluster Size"] = pd.Series(self.clustering_parameters.min_cluster_size)
        result["Cluster Metric"] = pd.Series(self.clustering_parameters.metric)
        result["Cluster Selection Method"] = pd.Series(self.clustering_parameters.cluster_selection_method)

        return result

    def reduce_dimensionality(self):
        model_reducer = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, metric=self.metric)
        reduced_embeddings = model_reducer.fit_transform(self.sentence_model.embeddings)
        return reduced_embeddings
