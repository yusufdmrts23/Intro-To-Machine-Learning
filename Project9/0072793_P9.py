import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spa
from numpy.linalg import eig


X = np.genfromtxt("hw09_data_set.csv", delimiter = ",")
N = np.shape(X)[0]
B = np.zeros((N,N))
S = 2
for i in range(N):
    for j in range(i+1,N):
        if np.sqrt( ( X[i][0] -X[j][0])**2 + ( X[i][1] -X[j][1])**2 ) < S:
            B[i][j] = 1
            B[j][i] = 1
            

plt.figure(figsize=(8, 8))

for i in range(N):
    for j in range(i+1,N):     
        if B[i][j] == 1:
            x_values = [X[i,0], X[j,0]]
            y_values = [X[i,1], X[j,1] ]
            plt.plot(x_values,y_values, color ="#606060")
            
plt.plot(X[:, 0], X[:, 1], ".", markersize = 10, color ='black')   
plt.show()


D = np.zeros((N,N))
D_inv = np.zeros((N,N))
L = D - B
I = np.zeros((N,N))
for i in range(N):
    D[i,i] = sum(B[i,:])

for i in range(N):
    D_inv[i,i] = np.sqrt( 1 / D[i,i])
    
for i in range(N):
    I[i,i] = 1
L_sym = I - np.matmul(D_inv, np.matmul(B,D_inv))
print(L_sym[0:5, 0:5])
K = 9 
R = 5
Eigen, Vector = eig(L_sym)
index = np.argsort(Eigen)[:R+1]
Z = np.zeros((N,R))
for i in range(R):
    Z[:,i] =  Vector[:,index[i+1] ]
print(Z[0:5, 0:5])
def update_centroids(memberships, X):
    if memberships is None:
        centroids =np.array([Z[242,:],Z[528,:],Z[570,:],Z[590,:],Z[648,:],Z[667,:],Z[774,:],Z[891,:],Z[955,:]])
    else:
        centroids = np.vstack([np.mean(X[memberships == k,:], axis = 0) for k in range(K)])
    return(centroids)

def update_memberships(centroids, X):
    # calculate distances between centroids and data points
    D = spa.distance_matrix(centroids, X)
    # find the nearest centroid for each data point
    memberships = np.argmin(D, axis = 0)
    return(memberships)

def plot_current_state(centroids, memberships, X):
    cluster_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a", "#b15928",
                               "#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6", "#ffff99"])
    if memberships is None:
        plt.plot(X[:,0], X[:,1], ".", markersize = 10, color = "black")
    else:
        for c in range(K):
            plt.plot(X[memberships == c, 0], X[memberships == c, 1], ".", markersize = 10,
                     color = cluster_colors[c])
    for c in range(K):
        plt.plot(centroids[c, 0], centroids[c, 1], "s", markersize = 12, 
                 markerfacecolor = cluster_colors[c], markeredgecolor = "black")
    plt.xlabel("x1")
    plt.ylabel("x2")
    
centroids = None
memberships = None
iteration = 1
while True:

    temp_centroids = centroids
    centroids = update_centroids(memberships, Z)
    if np.alltrue(centroids == temp_centroids):
        break

    temp_memberships = memberships
    memberships = update_memberships(centroids, Z)
    if np.alltrue(memberships == temp_memberships):
        break
    
    iteration = iteration + 1

centroids = update_centroids(memberships, X) 
plt.figure(figsize=(8, 8))  
plot_current_state(centroids, memberships, X)
plt.show()
