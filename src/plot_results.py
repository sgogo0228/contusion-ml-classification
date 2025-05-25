import matplotlib.pyplot as plt
import numpy as np

# show the result of dimentionality reduction methods as a function of feature numbers
fig = plt.figure(figsize=(6,4), edgecolor='blue')
classfier = 'LGBM'
result_dir = rf'D:\DaMing\contusion_classification'

data = np.load(rf'{result_dir}\{classfier}_pca_5rep_val_acc.npy')
plt.plot(5*np.arange(1, data.shape[1]+1), data.mean(axis=0), marker='o', linewidth=3, color='black')

data = np.load(rf'{result_dir}\{classfier}_mrmr_5rep_val_acc.npy')
plt.plot(5*np.arange(1, data.shape[1]+1), data.mean(axis=0), marker='o', linewidth=3, color='blue')

data = np.load(rf'{result_dir}\{classfier}_relief_5rep_val_acc.npy')
plt.plot(5*np.arange(1, data.shape[1]+1), data.mean(axis=0), marker='o', linewidth=3, color='red')

plt.xticks(fontsize=16, fontweight='bold', fontfamily='Times New Roman')
plt.yticks(fontsize=16, fontweight='bold', fontfamily='Times New Roman')
plt.legend(labels=['pca', 'mrmr', 'relief'], loc='lower right', edgecolor='None', prop={'family':'Times New Roman', 'size':14})
ax = fig.gca()
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)
plt.show()