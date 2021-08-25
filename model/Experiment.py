import umap
import hdbscan
import pandas as pd


class Experiment:
    def __init__(self, sentence_model, umap_parameters, clustering_parameters):
        self.sentence_model = sentence_model
        self.n_neighbors = umap_parameters.n_neighbors
        self.n_components = umap_parameters.n_components
        self.umap_metric = umap_parameters.metric
        self.min_cluster_size = clustering_parameters.min_cluster_size
        self.cluster_metric = clustering_parameters.metric
        self.cluster_selection_method = clustering_parameters.cluster_selection_method

    def get_result(self):
        result = pd.DataFrame()
        result["Model Name"] = pd.Series(self.sentence_model.model_name)

        result["Umap N Neighbors"] = pd.Series(self.n_neighbors)
        result["Umap N Components"] = pd.Series(self.n_components)
        result["Umap Metric"] = pd.Series(self.umap_metric)

        result["Cluster Size"] = pd.Series(self.min_cluster_size)
        result["Cluster Metric"] = pd.Series(self.cluster_metric)
        result["Cluster Selection Method"] = pd.Series(self.cluster_selection_method)

        try:
            cluster = self.clusterize()
            number_of_clusters = len(set(cluster.labels_))
        except:
            # 0 means no clusters generated, i.e. error
            number_of_clusters = 0

        result["Number of clusters"] = pd.Series(number_of_clusters)

        return result

    def reduce_dimensionality(self):
        model_reducer = umap.UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, metric=self.umap_metric)
        reduced_embeddings = model_reducer.fit_transform(self.sentence_model.embeddings)
        return reduced_embeddings


    def clusterize(self):
        reduced_embeddings = self.reduce_dimensionality()
        cluster = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                  metric=self.cluster_metric,
                                  cluster_selection_method=self.cluster_selection_method).fit(reduced_embeddings)
        return cluster


    def to_string(self):
        return "Sentence model: {}, " \
               "Umap components: {}, " \
               "Umap neighbors: {}, " \
               "Umap metric: {}, " \
               "Cluster size: {}, " \
               "Cluster metric: {}, " \
               "Cluster selection method: {}".format(self.sentence_model.model_name,
                                                     self.n_components,
                                                     self.n_neighbors,
                                                     self.umap_metric,
                                                     self.min_cluster_size,
                                                     self.cluster_metric,
                                                     self.cluster_selection_method
                                                     )