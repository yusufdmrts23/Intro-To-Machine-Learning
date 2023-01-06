# Intro-To-Machine-Learning
My Machine Learning Projects at Koç University

Yusuf Demirtaş 0072793

Short Descriptions of Project

PROJECT 1)

This  assignment involves implementing a classification algorithm in Python, using a confusion matrix to analyze the accuracy of the classification on the training set, and visualizing the decision boundaries of the classification method along with any misclassified data points. The parameters for the classification rule will be estimated from the training data.

PROJECT 2)

This assignment involves implementing a Naive Bayes classifier in Python for a multivariate classification dataset of 195 handwritten letters that belong to one of 5 classes (A, B, C, D, or E). The letters are 20 pixels by 16 pixels in size, and there are 39 data points from each class. We provided with two data files, and are asked to calculate confusion matrices for the training and test sets using the classification rule developed from the estimated parameters. The confusion matrices should be in a specific format.

PROJECT 3)

This assignment involves implementing a multiclass classification algorithm using discrimination by regression in Python. The algorithm will be modified to use K sigmoid functions to generate predicted class probabilities and the sum squared errors as the error function to minimize, rather than the negative log-likelihood and the softmax function. The classification dataset consists of 195 handwritten letters belonging to 5 classes (A, B, C, D, or E), with 39 data points in each class. The data will be divided into a training set with 25 images from each class and a test set with the remaining 14 images from each class. The goal is to learn a discrimination by regression algorithm for this multiclass classification problem, using the specified learning parameters, and to calculate a confusion matrix for the training set. The confusion matrix should be in a specific format.

PROJECT 4)

The assignment involves implementing three nonparametric regression algorithms in Python for a univariate regression dataset. The dataset consists of 180 training data points and 180 test data points. The algorithms to be implemented are a regressogram, a running mean smoother, and a kernel smoother. The bin width parameter will be set to 0.1 for the regressogram and the running mean smoother, and 0.02 for the kernel smoother. The origin parameter will be set to 0.0 for the regressogram. The goal is to draw the training and test data points along with the regressogram, running mean smoother, and kernel smoother in separate figures, and to calculate the root mean squared error (RMSE) for the test data points for each algorithm. The RMSE values and bin width parameters will be printed in a specific format.

PROJECT 5)

The assignment involves implementing a decision tree regression algorithm in Python for a univariate regression dataset. The dataset consists of 180 training data points and 180 test data points. The algorithm will use a pre-pruning rule where nodes with P or fewer data points are converted to terminal nodes and not split further, where P is a user-defined parameter. The goal is to learn a decision tree by setting the pre-pruning parameter P to 30, and to draw the training and test data points along with the fit in a single figure. The root mean squared error (RMSE) will be calculated for both the training and test data points. The RMSE values will be printed in a specific format. Additionally, decision trees will be learned by setting the pre-pruning parameter P to a range of values from 10 to 50, and the RMSE for the training and test data points will be plotted as a function of P in a single figure.

PROJECT 6)

Assignment involves implementing a support vector machine (SVM) classifier in Python for a binary classification dataset of 2000 clothing images that belong to either the "trouser" or "sandal" class. The images are 28 pixels by 28 pixels in size, and are provided in two data files: one containing the images, and the other containing the corresponding class labels. The data will be divided into a training set with 1000 images and a test set with the remaining 1000 images. Each grayscale image will be represented by a color histogram with 64 bins, and the color histograms will be calculated as the ratios of pixel values in each bin. The histogram intersection kernel will be used to calculate the similarities between input images, and the training and test kernel matrices will be calculated using the histograms. An SVM classifier will be trained on the training kernel matrix, using a regularization parameter of C=10, and confusion matrices will be calculated for the training and test data points using the training and test kernel matrices. The accuracy for the training and test data points will also be plotted as a function of C for a range of values of C.

PROJECT 7)

This assignment involves implementing the linear discriminant analysis (LDA) algorithm in Python for a multiclass classification dataset of 4000 clothing images that belong to one of 10 classes: "t-shirt/top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", and "ankle boot". The images are 28 pixels by 28 pixels in size, and are provided in two data files. The data will be divided into a training set with 2000 images and a test set with the remaining 2000 images. The LDA algorithm will be used to calculate the S1 and S2 matrices, the largest nine eigenvalues and corresponding eigenvectors of the S2 matrix, and to project the training and test data points onto a two-dimensional and nine-dimensional subspace. A k-nearest neighbor classifier will be trained with k=11 on the nine-dimensional projections, and confusion matrices will be calculated for the training and test data points. The confusion matrices should be in a specific format.

PROJECT 8)

This assignment involves implementing an expectation-maximization (EM) clustering algorithm in Python for a two-dimensional dataset of 1000 data points generated from nine bivariate Gaussian densities. The mean vectors and initial centroids for the EM algorithm are provided in a separate data file. The EM algorithm will be initialized by estimating the initial covariance matrices and prior probabilities, and then run for 100 iterations. The mean vectors found by the EM algorithm will be reported, and the resulting clustering will be plotted along with the original Gaussian densities and the Gaussian densities found by the EM algorithm. The original and found Gaussian densities will be plotted 

PROJECT 9)

The assignment involves implementing a spectral clustering algorithm in Python for a two-dimensional dataset of 1000 data points generated from nine bivariate Gaussian densities. The connectivity matrix will be constructed by considering pairs of data points with a distance less than 2.0 to be connected. The matrix will be visualized by drawing a line between connected data points. The matrices D and L will be calculated, and the normalized Laplacian matrix will be found using the formula L_norm = I - D^(-1/2) * B * D^(-1/2). The R=5 eigenvectors corresponding to the R smallest eigenvalues of the normalized Laplacian matrix will be found, and used to construct the matrix Z. K-means clustering will be run on the Z matrix to find K=5 clusters, using the rows of the Z matrix as initial centroids. The resulting clustering will be plotted, with each cluster colored differently.

