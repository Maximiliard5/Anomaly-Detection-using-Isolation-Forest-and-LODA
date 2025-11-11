from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score,roc_auc_score


#Ex1
#1.1

X, y = make_blobs(n_samples=500, n_features=2,
                              centers=[[0,0]], cluster_std=1,
                              shuffle=True,
                              random_state=None, return_centers=False)

plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Blob clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

#1.2
random_vectors=np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 5)
random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)

projections=X@random_vectors.T

num_bins=30
histograms = []
for i in range(random_vectors.shape[0]):
    hist, bin_edges = np.histogram(
        projections[:, i],
        bins=num_bins,
        range=(-12, 12),
        density=True
    )
    bin_width=bin_edges[1]-bin_edges[0]
    hist=hist/np.sum(hist*bin_width)

    histograms.append((hist, bin_edges))

plt.figure(figsize=(10, 8))

for i in range(len(histograms)):
    hist, bin_edges = histograms[i]
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.subplot(3, 2, i + 1)
    plt.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], alpha=0.7)
    plt.title(f'Projection {i + 1}')
    plt.xlabel('Projected value')
    plt.ylabel('Density')
    plt.grid(True)

plt.tight_layout()
plt.show()

probs=np.zeros_like(projections)

for i, (hist, bins) in enumerate(histograms):

    bin_idx=np.searchsorted(bins, projections[:, i])-1
    bin_idx=np.clip(bin_idx, 0, len(hist) - 1)

    probs[:,i]=hist[bin_idx]

eps=1e-8
scores=-np.mean(np.log(probs + eps), axis=1)
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='inferno')
plt.colorbar(label='Anomaly Score')
plt.title('Anomaly Scores')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

#1.3
X_test=np.random.uniform(-3,3, (500,2))
print(X_test.shape)

test_projections=X_test@random_vectors.T

test_probs=np.zeros_like(test_projections)

for i,(hist, bins) in enumerate(histograms):
    bin_idx=np.searchsorted(bins, test_projections[:,i]) - 1
    bin_idx=np.clip(bin_idx, 0,len(hist)-1)
    test_probs[:,i]=hist[bin_idx]

eps=1e-8
test_scores=-np.mean(np.log(test_probs + eps),axis=1)

plt.figure(figsize=(8,6))
plt.scatter(X_test[:,0], X_test[:,1],c=test_scores, cmap='inferno')
plt.colorbar(label='Anomaly Score')
plt.title('Anomaly Scores on Uniform Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

#1.4
bin_settings=[10,30,60, 100]  # try different bin counts

plt.figure(figsize=(12, 10))
for j, num_bins in enumerate(bin_settings):
    histograms=[]
    for i in range(random_vectors.shape[0]):
        hist, bin_edges=np.histogram(
            projections[:,i],
            bins=num_bins,
            range=(-12,12),
            density=True
        )
        # Normalize (optional but consistent)
        bin_width=bin_edges[1]-bin_edges[0]
        hist=hist/np.sum(hist*bin_width)
        histograms.append((hist,bin_edges))

    # Project and score test data
    test_projections=X_test @ random_vectors.T
    test_probs=np.zeros_like(test_projections)

    for i,(hist, bins) in enumerate(histograms):
        bin_idx=np.searchsorted(bins, test_projections[:,i])-1
        bin_idx=np.clip(bin_idx, 0,len(hist)-1)
        test_probs[:,i]=hist[bin_idx]

    eps=1e-8
    test_scores = -np.mean(np.log(test_probs + eps),axis=1)

    plt.subplot(2,2,j+1)
    plt.scatter(X_test[:, 0],X_test[:,1],c=test_scores,cmap='inferno')
    plt.title(f'Anomaly Map (bins={num_bins})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly Score')
    plt.grid(True)

plt.tight_layout()
plt.show()

#Ex2
#2.1
X,y=make_blobs(n_samples=[500,500], n_features=2,
               centers=((10,0),(0,10)),
               cluster_std=1)

plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Blob clusters ex2')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

#2.2

contamination=0.02
model=IForest(contamination=contamination)
model.fit(X)

test_data=np.random.uniform(-10,20,(1000,2))

model_predictions=model.decision_function(test_data)

#2.3
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    test_data[:, 0], test_data[:, 1],
    c=model_predictions, cmap='inferno'
)
plt.colorbar(scatter, label='Predicted Label (0=inlier, 1=outlier)')
plt.title('Isolation Forest Predictions on Test Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()

#2.4
contamination=0.02
X,y=make_blobs(n_samples=[500,500], n_features=2,
               centers=((10,0),(0,10)),cluster_std=1)

test_data=np.random.uniform(-10,20,(1000,2))

models ={
    "IForest": IForest(contamination=contamination),
    "DIF": DIF(contamination=contamination),
    "LODA": LODA(contamination=contamination),
}
scores={}

for name, model in models.items():
    model.fit(X)
    scores[name]=model.decision_function(test_data)

fig,axes=plt.subplots(2,2,figsize=(12,10))
axes=axes.ravel()

for i, (name, score) in enumerate(scores.items()):
    sc=axes[i].scatter(test_data[:,0],test_data[:,1],
                         c=score, cmap='inferno', s=15)
    axes[i].set_title(f'{name} Anomaly Scores')
    axes[i].set_xlabel('Feature 1')
    axes[i].set_ylabel('Feature 2')
    axes[i].grid(True)
    fig.colorbar(sc, ax=axes[i], label='Anomaly Score')
if len(scores)<len(axes):
    axes[-1].axis('off')
plt.show()

#2.5
hidden_configs=[(8,),(32,16),(64,32,16)]

test_data=np.random.uniform(-10, 20, (1000, 2))
fig, axes=plt.subplots(1, len(hidden_configs), figsize=(16, 5))

for ax,config in zip(axes,hidden_configs):
    model=DIF(hidden_neurons=config, contamination=0.02)
    model.fit(X)
    scores=model.decision_function(test_data)

    sc=ax.scatter(test_data[:, 0], test_data[:, 1], c=scores, cmap='inferno', s=15)
    ax.set_title(f'DIF hidden_neurons={config}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    fig.colorbar(sc,ax=ax,label='Anomaly Score')
    ax.grid(True)

plt.show()

bin_settings=[10,50,100]
fig,axes=plt.subplots(1, len(bin_settings), figsize=(16, 5))

for ax,bins in zip(axes,bin_settings):
    model=LODA(n_bins=bins,contamination=0.02)
    model.fit(X)
    scores=model.decision_function(test_data)

    sc=ax.scatter(test_data[:,0],test_data[:,1],c=scores, cmap='inferno', s=15)
    ax.set_title(f'LODA n_bins={bins}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    fig.colorbar(sc, ax=ax, label='Anomaly Score')
    ax.grid(True)

plt.show()

#2.6

X,y=make_blobs(
    n_samples=[500, 500],
    n_features=3,
    centers=((0,10,0),(10,0,10)),
    cluster_std=1,
)

test_data=np.random.uniform(-10, 20, (1000, 3))

contamination=0.02
models={
    "IForest": IForest(contamination=contamination),
    "DIF": DIF(contamination=contamination),
    "LODA": LODA(contamination=contamination)
}

# --- Train and Score ---
scores = {}
for name,model in models.items():
    model.fit(X)
    scores[name]=model.decision_function(test_data)

fig=plt.figure(figsize=(18, 6))
titles=list(scores.keys())

for i,name in enumerate(titles,start=1):
    ax=fig.add_subplot(1,3,i,projection='3d')
    sc=ax.scatter(
        test_data[:,0],test_data[:,1],test_data[:,2],
        c=scores[name], cmap='inferno', s=15
    )
    ax.set_title(f'{name} Anomaly Scores (3D)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(sc,ax=ax,shrink=0.6,label='Anomaly Score')

plt.show()

#3.1
data=loadmat('shuttle.mat')
print(data.keys())

X=data['X']
y=data['y'].ravel()

#3.2

models = {
    "IForest":IForest(contamination=0.02),
    "LODA":LODA(contamination=0.02),
    "DIF":DIF(contamination=0.02)
}

n_splits=10
results={name:{"BA":[],"ROC_AUC":[]} for name in models}

for split in range(n_splits):
    print(f'Starting split: {split+1}/{n_splits}')
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,stratify=y)

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)

    for name,model in models.items():
        model.fit(X_train)

        scores=model.decision_function(X_test)
        preds=model.predict(X_test)

        ba=balanced_accuracy_score(y_test, preds)
        roc_auc=roc_auc_score(y_test, scores)

        results[name]["BA"].append(ba)
        results[name]["ROC_AUC"].append(roc_auc)

print("Average performance over 10 splits")
for name in models:
    mean_ba=np.mean(results[name]["BA"])
    std_ba=np.std(results[name]["BA"])
    mean_auc=np.mean(results[name]["ROC_AUC"])
    std_auc=np.std(results[name]["ROC_AUC"])

    print(f"\n{name}:")
    print(f"Balanced Accuracy (mean +- std): {mean_ba:.4f} +- {std_ba:.4f}")
    print(f"ROC AUC (mean +- std): {mean_auc:.4f} +- {std_auc:.4f}")