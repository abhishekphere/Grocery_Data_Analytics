"""
    This program implements PCA and reduces the dimensions of the data to 3D.
"""
import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

data_og = pandas.read_csv("SHOPPING_CART.csv", encoding='cp1252')  # reads the data from csv
data = data_og.drop('ID', 1)                                                   # drops the column ID from data

correlation = numpy.around(numpy.corrcoef(data.T), decimals=2)                 # Finds the correlation of the data
print("Correlation matrix:")
print(correlation)

# part A
# Performs covariance of the data
cov = numpy.cov(data.T)                   # covariance of the entire data
print("Cov matrix:")
print(cov)


# Part B
# Calculates eigenvectors and eigenvalues of the covariance matrix
eig_values, eig_vector = numpy.linalg.eig(cov)
eig_vector = numpy.asarray([eig_vector[:,i] for i in range(len(eig_vector))])

print("eig_values: ", eig_values.shape)
print("eig_values : ", eig_values)

print("eig_vector: ", eig_vector.shape)
print("eig_vector: ", eig_vector)

eigen_dict = {}                                 # Keeps track of eigenvalues and their corresponding vectors
for i in range(len(eig_values)):
    eigen_dict[eig_values[i]] = eig_vector[i]


# Part C
# Sorts the eigenvalues from highest absolute value to lowest absolute value.
eig_values = sorted(eig_values, key=abs, reverse=True)
print("Eigen values:")
print(eig_values)


# Part E
# Prints first five eigen values
first_five_eigen_vectors = []                   # Keeps track of first five eigen values in descending order
for eig_val in range(5):
    first_five_eigen_vectors.append(numpy.around(eigen_dict[eig_values[eig_val]], decimals=1, out=None).tolist())
print("First Five eigen vactors: ")
print(first_five_eigen_vectors)


# Part F
# Project the original Agglomeration data onto eigenvectors
first_three_eigen_vectors = numpy.asarray(first_five_eigen_vectors[:3])
print("first_three_eigen_vectors:")
print(first_three_eigen_vectors)

projection = numpy.dot(first_three_eigen_vectors, data.T)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(projection[0], projection[1], projection[2], marker='o')
plt.show()


# Part G
# Performs k means clustering and finds six clusters
kmeans = KMeans(n_clusters=6, random_state=0).fit(projection.T)
print("Labels in kmeans:")
print(kmeans.labels_)
print("kmeans cluster centers:")
print(kmeans.cluster_centers_)


# Saves the projected data to csv file
df = pandas.DataFrame(projection.T)
df.insert(0, 'ID', data_og['ID'], True)
df.to_csv("projected_data.csv", index=False)

