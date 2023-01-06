import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from scipy.stats import multivariate_normal

X = np.genfromtxt("hw08_data_set.csv", delimiter=",")
initial_centroids = np.genfromtxt("hw08_initial_centroids.csv", delimiter=",")

N = np.shape(X)[0]
K = np.shape(initial_centroids)[0]
D = np.shape(X)[1]

r_means = np.array([[+5.0, +5.0],[-5.0, +5.0],[-5.0, -5.0],[+5.0, -5.0],[+5.0, +0.0],[+0.0, +5.0],[-5.0, +0.0],[+0.0, -5.0],[+0.0, +0.0]])
r_covariances = np.array([[[+0.8, -0.6], [-0.6, +0.8]],[[+0.8, +0.6], [+0.6, +0.8]],[[+0.8, -0.6], [-0.6, +0.8]],[[+0.8, +0.6], [+0.6, +0.8]],[[+0.2, +0.0], [+0.0, +1.2]],[[+1.2, +0.0], [+0.0, +0.2]],[[+0.2, +0.0], [+0.0, +1.2]],[[+1.2, +0.0], [+0.0, +0.2]],[[+1.6, +0.0], [+0.0, +1.6]]])


def update_memberships(centroids, data):
    d = spa.distance_matrix(centroids, data)
    memberships = np.argmin(d, axis=0)
    return memberships


def find_priors(data, memberships):
    return [data[memberships == c].shape[0] / N for c in range(K)]

def find_covariance_matrix(new_data, centroid):
    sum_matrix = np.zeros((2, 2))
    for i in range(new_data.shape[0]):
        covariance = np.matmul((new_data[i] - centroid).reshape(2, 1),(new_data[i] - centroid).reshape(1, 2))
        sum_matrix += covariance
    return sum_matrix / new_data.shape[0]


def find_covariance_matrices(data, memberships, centroids):
    return [find_covariance_matrix(data[memberships == k], centroids[k]) for k in range(K)]

centroids = initial_centroids
memberships = update_memberships(centroids, X)
covariances = find_covariance_matrices(X, memberships, centroids)

def find_pos_probs(centroids, covariances, priors):
    return [multivariate_normal(centroids[k], covariances[k]).pdf(X) * priors[k] for k in range(K)]

def update_centroids(data, memberships):
    return np.vstack([np.matmul(memberships[k], data) / np.sum(memberships[k], axis=0) for k in range(K)])

def update_covariance(data, membership, mean):
    sum_matrix = np.zeros((2, 2))
    for i in range(N):
        covariance = np.matmul((data[i] - mean).reshape(2, 1),(data[i] - mean).reshape(1, 2)) * membership[i]
        sum_matrix += covariance
    return sum_matrix / np.sum(membership, axis=0)


def update_covariances(data, memberships, means):
    return [update_covariance(data, memberships[k], means[k]) for k in range(K)]

priors = find_priors(X, memberships)

def update_priors(memberships):
    return np.vstack([np.sum(memberships[k], axis=0) / N for k in range(K)])







for i in range(100):
    post_probabs = find_pos_probs(centroids, covariances, priors)
    memberships_2 = np.vstack(
        [post_probabs[k] / np.sum(post_probabs, axis=0) for k in range(K)])
    centroids = update_centroids(X, memberships_2)
    covariances = update_covariances(X, memberships_2, centroids)
    priors = update_priors(memberships_2)


posterior_probabilities = find_pos_probs(centroids, covariances, priors)
memberships_2 = np.vstack([posterior_probabilities[k] / np.sum(post_probabs, axis=0) for k in range(K)])
memberships = np.argmax(memberships_2, axis=0)
print('print(means)')
print(centroids)




colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99"])
plt.figure(figsize = (12, 6))  
x1, x2 = np.mgrid[-6:+6:.05, -6:+6:.05] # grid for plt.contour
pos = np.dstack((x1, x2))


for c in range(K):
    predicted_classes = multivariate_normal(centroids[c], covariances[c]).pdf(pos)
    real_classes = multivariate_normal(r_means[c], r_covariances[c]).pdf(pos)
    plt.contour(x1, x2, predicted_classes, levels=1, colors=colors[c])
    plt.contour(x1, x2, real_classes, levels=1, linestyles="dashed", colors="k")
    plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize=10,
             color=colors[c])

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()