# cSVD
Clustered Singular Value Decomposition (cSVD) is a matrix decomposition that factorizes a given dataset using its Singular Value Decomposition (SVD) and reconstructs the features by projection matrices of clustered subsets. In this method, k-means clustering is used to label the whole data into several subsets in a low-dimensional space.

<p align="center">
<img width="700" alt="Screenshot 2023-02-04 at 21 17 54" src="https://user-images.githubusercontent.com/52735725/216788318-8ec03682-41de-4413-9255-af8bccdf8aab.png">



## 1. SVD
$\rho=Vv$, $v \in D$ (a given dataset)

$v \approx \tilde v=V^\top\rho$

## 2. cSVD
$\rho=Vv$

$v \approx \tilde v=W_l\top(W_lV^\top\rho)$

where $\rho=W_lw$, $w \in D_l$ satisfying $D=\cup_{l=1}^k D_l$, $\cap_{l=1}^k  D_l \neq \emptyset$, $l=1,2,\cdots , k.$ and  $k$ is the number of clusters.

## 3. Getting started
```python
import csvd

  ...

cs = csvd.ClusteredSVD(data, n_svd_components=n_svd_components, num_clusters=num_clusters)
V = cs.V
W = cs.wlist
rec_svd = (V.T @ cs.rho).squeeze(-1)
rec_csvd = cs.rec

  ...
```
#### Parameters
----------
- data: B x N dataset matrix (B: number of instances, N: number of features)
- n_svd_components: reduced parameter dimension
- num_clusters: number of clusters

#### Returns
-------
- V: a projection matrix V obtained by SVD with the whole data
- rho: reduced dimensional data
- labels: clustering labels
- wlist: a list of k projection matrices Ws obtained by SVD with each clustered dataset
        
## 4. Test
### 4.1. Sklearn Datasets 
(#instances, #features), n_components, num_clusters
- D1. wine dataset: (178, 13), n_components=5, num_clusters=4
- D2. breast cancer dataset: (569, 30), n_components=5, num_clusters=3
- D3. mnist dataset: (1797, 64), n_components=20, num_clusters=10
- D4. covertype dataset: (581012, 54), n_components=5, num_clusters=10

### 4.2. Reconstruction Error (MSE)
| type | D1 | D2 | D3 | D4 |
| ---  | ---| ---| ---| ---| 
| SVD  | 0.1745 | 0.2760 |1.9888 | 78.5639 |
| cSVD | **0.1705** | **0.1863** | **1.6580** | **73.0463** | 

## 5. Reference
> Yongho Kim and Jan Heiland (2023), *Convolutional Autoencoders, Clustering, and POD for Low-dimensional Parametrization of Navier-Stokes Equations*


