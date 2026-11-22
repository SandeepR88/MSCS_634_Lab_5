# MSCS_634_Lab_5: Hierarchical and DBSCAN Clustering

**Course:** Advanced Big Data and Data Mining (MSCS-634-B01)  
**Lab:** Hierarchical and DBSCAN Clustering  
**Dataset:** Wine dataset from `sklearn.datasets`

## 1. Purpose of the Lab

The purpose of this lab was to apply and compare two unsupervised machine-learning algorithms—Agglomerative Hierarchical Clustering and DBSCAN—using the Wine dataset. The work involved exploring and standardizing the dataset, testing different clustering configurations, visualizing the resulting clusters, and evaluating clustering quality using the Silhouette Score, Homogeneity Score, and Completeness Score.

The Wine dataset contains **178 observations and 13 numerical features**. All 13 standardized features were used for clustering. Principal Component Analysis (PCA) was used to project the data into two dimensions for visualization. The first two principal components explained approximately **55.41%** of the dataset's variance.

## 2. Key Results and Insights

### Agglomerative Hierarchical Clustering

Three cluster counts were tested: 2, 3, and 4. The results were:

| Number of clusters | Silhouette Score | Homogeneity | Completeness |
|---:|---:|---:|---:|
| 2 | 0.2670 | 0.4441 | 0.7745 |
| 3 | 0.2774 | 0.7904 | 0.7825 |
| 4 | 0.2258 | 0.7904 | 0.6490 |

The **three-cluster solution** provided the strongest overall balance among the tested hierarchical configurations. It achieved the highest silhouette score (0.2774) and high homogeneity (0.7904) and completeness (0.7825). The four-cluster solution did not improve the silhouette score and had lower completeness. The dendrogram provided a visual representation of the hierarchical merging process and helped illustrate how clusters were combined at increasing linkage distances.

### DBSCAN

DBSCAN was tested with several combinations of `eps` and `min_samples`. The results were:

| eps | min_samples | Clusters found | Noise points | Silhouette* | Homogeneity | Completeness |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 5 | 0 | 178 | N/A | 0.0000 | 1.0000 |
| 2.0 | 5 | 5 | 85 | 0.2405 | 0.3624 | 0.3247 |
| 2.5 | 5 | 1 | 24 | N/A | 0.0576 | 0.1581 |
| 3.0 | 5 | 1 | 11 | N/A | 0.0309 | 0.1446 |
| 2.0 | 10 | 1 | 139 | N/A | 0.2887 | 0.5963 |
| 2.5 | 10 | 2 | 32 | 0.3228 | 0.5045 | 0.5604 |

\*DBSCAN silhouette scores were calculated only on non-noise observations and only when at least two non-noise clusters were present. N/A indicates that a valid silhouette score could not be calculated because the non-noise data did not contain at least two clusters.

The strongest DBSCAN configuration among those tested was **`eps=2.5` and `min_samples=10`**. It identified two clusters and 32 noise observations, with a silhouette score of 0.3228, homogeneity of 0.5045, and completeness of 0.5604.

The parameter experiments showed that:
- With `eps=1.5` and `min_samples=5`, all 178 observations were classified as noise.
- Increasing `eps` generally allowed more observations to connect, reducing the number of noise points. However, larger neighborhoods also caused some configurations to merge into a single cluster.
- Increasing `min_samples` made the density requirement stricter. For example, `eps=2.0` and `min_samples=10` classified 139 observations as noise.
- The combination `eps=2.5` and `min_samples=10` produced two clusters and a more balanced result than the other tested DBSCAN configurations.

### Comparison of the Algorithms

Hierarchical Clustering with three clusters aligned more closely with the known Wine cultivar labels than the tested DBSCAN configurations. Its homogeneity (0.7904) and completeness (0.7825) were higher than the best DBSCAN configuration's scores (0.5045 and 0.5604, respectively).

DBSCAN's best silhouette score (0.3228) was higher than the hierarchical three-cluster score (0.2774). However, the DBSCAN score was calculated after excluding its 32 noise observations, whereas the hierarchical score included all observations. Therefore, the two silhouette scores should not be treated as a perfectly like-for-like comparison.

Overall, Hierarchical Clustering provided the strongest alignment with the known cultivar categories in this experiment. DBSCAN offered the additional ability to identify observations that did not belong to sufficiently dense groups.

## 3. Challenges and Decisions

1. **Different feature scales:** The Wine dataset contains 13 features with different measurement ranges. `StandardScaler` was applied before clustering so that features with larger numeric values would not dominate distance calculations.
2. **Visualizing high-dimensional data:** Since the clustering used all 13 features, PCA was used to display the cluster assignments in two dimensions. The PCA plots are visual projections and do not represent the full 13-dimensional feature space.
3. **Selecting hierarchical cluster counts:** The notebook tested 2, 3, and 4 clusters and compared their silhouette, homogeneity, and completeness scores rather than selecting a cluster count arbitrarily.
4. **Tuning DBSCAN:** DBSCAN's output changed substantially with `eps` and `min_samples`. Several combinations were tested to observe changes in cluster formation and noise detection.
5. **Undefined silhouette scores:** A silhouette score cannot be calculated when the evaluated observations contain fewer than two clusters. The notebook reports N/A (or `NaN`) for those DBSCAN configurations rather than presenting an invalid score.
6. **Evaluating against known categories:** The dataset's cultivar labels were used only to calculate homogeneity and completeness. They were not provided to either clustering algorithm during model fitting.

## 4. Conclusion

The experiment showed that clustering outcomes depend on preprocessing, algorithm selection, and parameter settings. Agglomerative Hierarchical Clustering with three clusters provided the strongest overall alignment with the known Wine cultivar categories. DBSCAN, using `eps=2.5` and `min_samples=10`, produced the best DBSCAN result among the tested configurations and identified 32 observations as noise. Hierarchical Clustering was more effective for recovering the known cultivar groupings in this experiment, while DBSCAN was useful for detecting dense regions and potential outliers.
