import numpy as np
import pandas as pd


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
tsne_results = TSNE(n_components=2, verbose=1, n_iter=500).fit_transform(X)
tsne_results.shape

df = pd.DataFrame()
df['tsne-2d-one'] = tsne_results[:,0]
df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot( 
    data=df,
    legend="full",
    alpha=0.3
)
plt.show()

sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)

$$$$


from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

iris = load_iris()
X_tsne = TSNE(n_components=2, verbose=1, n_iter=500).fit_transform(iris.data)
X_pca = PCA().fit_transform(iris.data)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target)
plt.subplot(122)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
