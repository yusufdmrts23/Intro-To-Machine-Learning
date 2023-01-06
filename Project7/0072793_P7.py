import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
from scipy import stats
from scipy.spatial import distance


images_data = np.genfromtxt("hw07_data_set_images.csv", delimiter = ",")
labels_data = np.genfromtxt("hw07_data_set_labels.csv", delimiter = ",")

X=images_data[:2000,:]
X_test=images_data[2000:,:]
Y=labels_data[:2000]
Y_test=labels_data[2000:]

classes = np.unique(Y)
K = len(classes) #no of classes
N = X.shape[0]
D = X.shape[1]


sample_mean = np.mean(X, axis=0)
SW = np.zeros((D,D))
SB = np.zeros((D, D))
for c in classes:
    X_c = X[Y == c]
    mean_c = np.mean(X_c, axis=0)
    SW += (X_c-mean_c).T.dot((X_c-mean_c))
    n_c = X_c.shape[0]
    mean_diff = (mean_c - sample_mean).reshape(D, 1)
    SB += n_c * (mean_diff).dot(mean_diff.T)
    
    
print(SW[0:4, 0:4])
print(SB[0:4, 0:4])

SWSB= np.linalg.inv(SW).dot(SB)

eigenvalues, eigenvectors = np.linalg.eig(SWSB)
values = np.real(eigenvalues)
eigenvectors = eigenvectors.T
vectors = np.real(eigenvectors)
print(values[0:9])



X_projected = np.dot(X, vectors[0 : 2].T)
                     
plt.figsize=(6,6)
point_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6"])
for c in range(K):
    plt.plot(X_projected[Y == c + 1, 0], X_projected[Y == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", 
"bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()

X_projected_test = np.dot(X_test, vectors[0 : 2].T)


plt.figsize=(6,6)
for c in range(K):
    plt.plot(X_projected_test[Y_test == c + 1, 0], X_projected_test[Y_test == c + 1, 1], marker = "o", markersize = 4, linestyle = "none", color = point_colors[c])
plt.legend(["t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", 
"bag", "ankle boot"],
           loc = "upper left", markerscale = 2)
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.xlabel("Component#1")
plt.ylabel("Component#2")
plt.show()



X_projected_9 = np.real(np.dot(X, vectors[0 : 9].T))
X_projected_test_9 = np.real(np.dot(X_test, vectors[0 : 9].T))

Y_hat = []
for i in range(N):
    test_set = X_projected_9[i,:]
    d1 = np.zeros(X_projected_9.shape[0])
    for j in range(N):
        d1[j] = distance.euclidean(test_set, X_projected_9[j, :])
    d2 = np.argsort(d1)[:11]

    for_stat = []
    for k in d2:
        for_stat.append(Y[k])
    pred= stats.mode(for_stat)[0]
    Y_hat.append(pred)

Train_predict = Y_hat
confusion_matrix = pd.crosstab(np.reshape(Train_predict, N), Y,
                               rownames = ["y_hat"], colnames = ["y_train"])
print(confusion_matrix)

Y_hat1 = []
Y_hat2 = []
for i in range(N):
    test_set1 = X_projected_9[i,:]
    test_set2 = X_projected_test_9 [i,:]
    d1 = np.zeros(X_projected_9.shape[0])
    d2 = np.zeros(X_projected_9.shape[0])
    for j in range(N):
        d1[j] = distance.euclidean(test_set1, X_projected_9[j, :])
        d2[j] = distance.euclidean(test_set2, X_projected_9[j, :])
    s_d1 = np.argsort(d1)[:11]
    s_d2 = np.argsort(d2)[:11]
    for_stat1 = []
    for_stat2 = []
    for k in s_d1:
        for_stat1.append(Y[k])
    for k in s_d2:
        for_stat2.append(Y[k])
    pred1= stats.mode(for_stat1)[0]
    pred2= stats.mode(for_stat2)[0]
    Y_hat1.append(pred1)
    Y_hat2.append(pred2)


Test_predict = Y_hat2
confusion_matrix_t = pd.crosstab(np.reshape(Test_predict, N), Y_test,
                               rownames = ["y_hat"], colnames = ["y_test"])
print(confusion_matrix_t)
