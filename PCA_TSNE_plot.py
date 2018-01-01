import numpy as np 
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # Principal Component Analysis module
from sklearn.manifold import TSNE # TSNE module
from sklearn.preprocessing import StandardScaler


########Figure 2############
data = pd.read_csv('data.csv')
# Convert the diagnosis column to numeric format
data['diagnosis'] = data['diagnosis'].factorize()[0]
# Fill all Null values with zero
data = data.fillna(value=0)
# Store the diagnosis column in a target object and then drop it
target = data['diagnosis']
data = data.drop('diagnosis', axis=1)
X = data.values

X_std = StandardScaler().fit_transform(X)
# Invoke the PCA method on the standardised data
pca = PCA(n_components=2)
#pca_2d = pca.fit_transform(X)
pca_2d_std = pca.fit_transform(X_std)
print 'ratio: ',pca.explained_variance_ratio_
# Invoke the TSNE method
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
#tsne_results = tsne.fit_transform(pca_2d)
tsne_results_std = tsne.fit_transform(pca_2d_std)


'''
#plot the PCA and TSNE for original data
plt.figure(figsize = (9,5))
plt.subplot(121)
plt.scatter(pca_2d[:,0],pca_2d[:,1], c = target, cmap = "summer", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('PCA Scatter Plot')
plt.subplot(122)
plt.scatter(tsne_results[:,0],tsne_results[:,1],  c = target, cmap = "summer", edgecolor = "None", alpha=0.35)
plt.colorbar()
plt.title('TSNE Scatter Plot')
plt.show()
'''


#plot the PCA and TSNE for std data
plt.figure(figsize = (9,5))
sns.set(style="darkgrid", palette="muted")
plt.subplot(121)
scatter1 = plt.scatter(pca_2d_std[:,0],pca_2d_std[:,1], c = target, cmap = "coolwarm", edgecolor = "None")
plt.colorbar()
plt.title('PCA Scatter Plot')
plt.subplot(122)
scatter2 = plt.scatter(tsne_results_std[:,0],tsne_results_std[:,1],  c = target, cmap = "coolwarm", edgecolor = "None")
plt.colorbar()
plt.title('TSNE Scatter Plot')
plt.show()
