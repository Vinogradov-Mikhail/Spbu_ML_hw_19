from scipy import sparse
from alive_progress import alive_bar
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class AffinityPropagation:
    def __init__(self):
        self.labels = None
        self.clust_loc = None


    def fit(self, S, max_iterations):
        S_matrix = S
        shape = S.shape

        A_matrix = sparse.csr_matrix(shape)

        R_matrix = sparse.lil_matrix(shape)
        with alive_bar(max_iterations) as bar:
            for i in range(max_iterations):
                print('Itertion ', i)

                sum_AS = A_matrix + S_matrix
                sum_AS_copy = sum_AS.copy()

                enum = np.arange(shape[0])
                max_ind = np.asarray(np.argmax(sum_AS, -1)).flatten()
                sum_AS_copy[enum, max_ind] = -np.inf

                first_maxs = np.asarray(sum_AS[enum, max_ind]).flatten()
                second_maxs = np.asarray(sum_AS_copy.max(-1).todense()).flatten()

                sum_AS = sum_AS.tocoo()

                for i, k, data in zip(sum_AS.row, sum_AS.col, sum_AS.data):
                    value = data - first_maxs[i]
                    if data == first_maxs[i]:
                        value = data - second_maxs[i]
                    R_matrix[i, k] = value

                A_new = sparse.lil_matrix(R_matrix.shape)
                R_copy = R_matrix.copy()
                a_k_k = np.zeros(R_matrix.shape[0])

                R_k_k = np.asarray(R_copy.diagonal()).flatten()
                R_copy.setdiag(0)
                R_max = R_copy.maximum(0).tocoo()
                maxs_sum = np.asarray(R_max.sum(0)).flatten()

                for i, k, data in zip(R_max.row, R_max.col, R_max.data):
                    if k != i:
                        a_k_k[k] += data
                    tmp = R_k_k[k] + maxs_sum[k] - data
                    A_new[i, k] = max(0, tmp)

                A_matrix = A_new.minimum(0).tocoo()
                A_matrix.setdiag(a_k_k)
                bar()

        print('Getting clusters...')
        idx = np.flatnonzero((np.asarray(A_matrix.diagonal()).flatten() + np.asarray(R_matrix.diagonal()).flatten()) > 0)
        clusters_count = len(idx)
        labels = np.asarray(S_matrix[:, idx].argmax(-1)).flatten()
        labels[idx] = np.arange(clusters_count)
        clusters = [np.where(labels == i)[0] for i in range(clusters_count)]
        self.labels = labels
        self.clusters = clusters
        i = 0
        tmp = []
        for cl in clusters:
            tmp.append([i, len(cl)])
            i = i + 1
        tmp.sort(key = lambda x: -x[1])
        print('Top 10:')
        print('No', ' cluster id', ' size of elements' )
        for p in range(10):
            print(p, ' ', tmp[p][0], '       ', tmp[p][1])
        print('Clusters count - ', clusters_count)

    def predict(self, checkins: pd.DataFrame):
        checkins["cluster"] = self.labels[checkins.user_id]
        cluster_locations = checkins.groupby(by="cluster")["location_id"].value_counts()
        return cluster_locations,list(checkins["location_id"].value_counts().index[:10])


    def get_top10_predict_for_user(self, clusters, checkins, user_id):
        target_cluster = self.labels[user_id]
        try:
            return list(clusters[target_cluster][:10].index)
        except IndexError:
            return list(checkins["location_id"].value_counts().index[:10])
        return None
