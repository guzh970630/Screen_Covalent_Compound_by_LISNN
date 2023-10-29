import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


x_drug = pd.read_csv("./data/100dim-drug.csv",sep=',', header=0)
x_drug = x_drug.values.tolist()
y_drug = pd.read_csv("./data/data1_shuffle.csv",sep=',')['number']
y_drug = y_drug.values.tolist()

drug_tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_drug_tsne = drug_tsne.fit_transform(x_drug)

fig, ax = plt.subplots(1,1,figsize=(40,40))
plt.cla()
for i in range(X_drug_tsne.shape[0]):
    plt.scatter(X_drug_tsne[i, 0], X_drug_tsne[i, 1], alpha = 1)
    plt.annotate(str(y_drug[i]), xy=(X_drug_tsne[i, 0], X_drug_tsne[i, 1]),xytext=(X_drug_tsne[i, 0] + 0.1, X_drug_tsne[i, 1] + 0.1))
plt.savefig("drug_tsne.jpg")
inset_ax = fig.add_axes([0.5, 0.1, 0.3, 0.3],facecolor="white")
for i in range(X_drug_tsne.shape[0]):
    plt.scatter(X_drug_tsne[i, 0], X_drug_tsne[i, 1], alpha = 1)
    plt.annotate(str(y_drug[i]), xy = (X_drug_tsne[i, 0], X_drug_tsne[i, 1]), xytext = (X_drug_tsne[i, 0]+0.1, X_drug_tsne[i, 1]+0.1))
inset_ax.set_xlim([18, 30])
inset_ax.set_ylim([13, 18])
inset_ax.grid()
mark_inset(ax, inset_ax, loc1=2, loc2=1, fc="none", ec='k', lw=1)
plt.savefig("drug_tsne_zoom.jpg")

print("finish!")