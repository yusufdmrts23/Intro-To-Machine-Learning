import math
import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("hw04_data_set_train.csv", delimiter = ",")
data_set_test = np.genfromtxt("hw04_data_set_test.csv", delimiter = ",")

X = data_set_train[:, 0]
Y = data_set_train[:, 1]
x_training = data_set_train[:,0]
y_training = data_set_train[:,1]
x_test = data_set_test[:,0]
y_test = data_set_test[:,1]
N = data_set_train.shape[0]

bin_width = 0.1
bin_origin = 0.0

minimum_value = min(np.min(X), bin_origin)
maximum_value = max(np.max(X), bin_origin)
data_interval = np.linspace(minimum_value, maximum_value, 1601)

left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

# Helper Functions
def Conditional(i, x):
    return (left_borders[i] < x) & (x <= right_borders[i])
def Kernel(u):
    return 1 / math.sqrt(2 * math.pi) * np.exp(-0.5 * (u ** 2))




# Regressogram
regressogram = np.array(
    [np.sum((Conditional(i, x_training)) * y_training) / np.sum(Conditional(i, x_training)) for i in range(len(left_borders))])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [regressogram[b], regressogram[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [regressogram[b], regressogram[b + 1]], "k-")
plt.show()


regressogram_rmse = 0
for i in range(len(left_borders)):
    for j in range(len(y_test)):
        if x_test[j] <= right_borders[i] and left_borders[i] <= x_test[j]:
            regressogram_rmse += (y_test[j] - regressogram[i]) ** 2

regressogram_rmse = math.sqrt(regressogram_rmse / len(y_test))
print("Regressogram => RMSE is {0} when h is {1}".format(regressogram_rmse, bin_width))


# Running Mean Smoother
mean_smooth = np.array([np.sum((abs((x - x_training) / bin_width)<= 0.5) * y_training)
                        / np.sum((abs((x - x_training) / bin_width)<= 0.5)) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
plt.plot(data_interval, mean_smooth, "k-")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
plt.plot(data_interval, mean_smooth, "k-")
plt.show()

mean_smoother_rmse = np.array([np.sum((abs((x - x_training) / bin_width)<= 0.5) * y_training)
                               / np.sum((abs((x - x_training) / bin_width)<= 0.5)) for x in x_test])
mean_smoother_error = np.sqrt(np.sum((y_test - mean_smoother_rmse) ** 2) / len(y_test))
print("Running Mean Smoother => RMSE is {0} when h is {1}".format(mean_smoother_error, bin_width))


# Kernel Smoother
bin_width = 0.02

kernel_smooth = np.array([np.sum(Kernel((x - x_training) / bin_width) * y_training)
                          / np.sum(Kernel((x - x_training) / bin_width)) for x in data_interval])

plt.figure(figsize=(10, 6))
plt.plot(x_training, y_training, "b.", markersize=10, label="Training")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
plt.plot(data_interval, kernel_smooth, "k-")
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(x_test, y_test, "r.", markersize=10, label="Test")
plt.xlabel("Eruption Time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.legend(["training", "test"], loc='upper left')
plt.plot(data_interval, kernel_smooth, "k-")
plt.show()

kernel_smoother_rmse = np.array([np.sum(Kernel((x - x_training) / bin_width) * y_training)
                                 / np.sum(Kernel((x - x_training) / bin_width)) for x in x_test])
kernel_smoother_error = np.sqrt(np.sum((y_test - kernel_smoother_rmse) ** 2) / len(y_test))
print("Kernel Smoother => RMSE is {0} when h is {1}".format(kernel_smoother_error, bin_width))



