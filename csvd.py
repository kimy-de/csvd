import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

class ClusteredSVD:
    def __init__(self, data, n_svd_components = 2, num_clusters = 3):
        """

        Parameters
        ----------
        data: B x N dataset matrix (B: number of instances, N: number of features)
        n_svd_components: reduced parameter dimension
        num_clusters: number of clusters

        Returns
        -------
        V: a projection matrix V obtained by SVD with the whole data
        rho: reduced dimensional data
        labels: clustering labels
        V_list: a list of k projection matrices Vs obtained by SVD with each clustered dataset

        """ 

        # Reduced feature dimension using SVD
        master_svd = TruncatedSVD(n_components=n_svd_components, random_state=42)
        master_svd.fit(data)
        self.V = master_svd.components_ 
        self.rho = self.V @ data.reshape(-1, data.shape[1], 1)

        # Clustering low-dimensional features
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.rho.squeeze(-1)) 
        self.labels = kmeans.labels_
        
        # Clustered SVD
        idx_list = []
        self.wlist = []
        rec_arr = np.array([]).reshape(0,data.shape[1])

        for i in range(num_clusters):
            idx = np.where(self.labels==i)[0]
            idx_list += idx.tolist()
            clustered_data = data[idx]
            clustered_rho = self.rho[idx]

            svd = TruncatedSVD(n_components=n_svd_components, random_state=42) # r x n
            svd.fit(clustered_data)
            W = svd.components_ # k x n
            self.wlist.append(W)
           
            recon = (W.T @(W @ self.V.T) @ clustered_rho).squeeze(-1)
            rec_arr = np.vstack([rec_arr, recon])

        self.wlist = np.array(self.wlist)

        idx_arr = np.array(idx_list)
        self.rec = self.reorder_idx(rec_arr, idx_arr)  


    def reorder_idx(self, A, idx):
        """

        This function reorganizes clustered datasets based on the original data index.

        Parameters
        ----------
        A: dataset including the entire clustered data
        idx: original data index

        Returns
        -------
        A: Reindexed dataset

        """ 

        A = np.concatenate([A,idx.reshape(-1,1)], axis=1)

        return A[(A[:,-1]).argsort()][:,:-1]
