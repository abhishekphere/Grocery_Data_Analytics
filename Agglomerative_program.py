"""
    This program implements PCA and reduces the dimensions of the data to 3D.
"""
import pandas
import math
import matplotlib.pyplot as plt
import copy
from scipy.cluster import hierarchy
import plotly.figure_factory as ff


og_data = pandas.read_csv("SHOPPING_CART.csv", encoding='cp1252')   # reads the original data from csv
projected_data = pandas.read_csv("projected_data.csv", encoding='cp1252')       # reads projected data from PCA
data = projected_data.drop('ID', 1)             # drops the column ID from data
data2 = og_data.drop('ID', 1)

# Draws dendogram on original data
fig = ff.create_dendrogram(data2)
fig.update_layout(width=800, height=500)
fig.show()

# Draws dendogram on projected data
fig = ff.create_dendrogram(data)
fig.update_layout(width=800, height=500)
fig.show()

list_of_rows = []                                  # stores data as list of rows
for row in projected_data.iterrows():
    list_of_rows.append(list(row[1]))

dict_of_clusters1 = {}              # Keeps track of cluster and their corresponding records
dict_of_records = {}                # Keeps track of records and their record ID
for cluster_index in range(1, len(list_of_rows) + 1):
    dict_of_clusters1[cluster_index] = [cluster_index]
    dict_of_records[cluster_index] = (list_of_rows[cluster_index-1])[1:]


def find_centroid(data_points):
    """
        This function finds the centroid of all the records in the cluster.

        :param   data_points:    data points in the cluster
        :return: center_of_mass: center of mass
    """
    center_of_mass = []                     # keeps track of center of mass for each attribute
    for col in range(len(data_points[0])):
        sum = 0
        for row in range(len(data_points)):
            sum += data_points[row][col]
        center_of_mass.append(sum/len(data_points))
    return center_of_mass

def find_distance(center_1, center_2):
    """
        This function finds the euclidean distance between two cluster centers.

        :param   center_1:    center of cluster 1
        :param   center_2:    center of cluster 2
        :return: distance:    distance
    """
    sum = 0
    for col in range(len(center_1)):
        sum += (center_1[col] - center_2[col])**2
    return math.sqrt(sum)

def get_datapoints(records):
    """
        This function finds the record data for specific record ids.

        :param   records:       list of record ids
        :return: data_points:   data_points
    """
    data_points = []
    for record_id in records:
        data_points.append(dict_of_records[record_id])
    return data_points


# Performs Agglomeration
index = 0                       # Keeps track of index
size_of_cluster = []            # Keeps track of size of the clusters
dict_of_six_clusters = {}       # Keeps track of the six clusters formed
while(True):

    min_distance = math.inf     # Keeps track of min distance between clusters

    # Finds minimum distance from one cluster to every other cluster
    for key1, val1 in dict_of_clusters1.items():
        for key2, val2 in dict_of_clusters1.items():
            # key 1 must never be equal to key 2
            if(key2 > key1):

                records_in_cluster1 = dict_of_clusters1[key1]
                records_in_cluster2 = dict_of_clusters1[key2]

                data_points_cluster1 = get_datapoints(records_in_cluster1)
                data_points_cluster2 = get_datapoints(records_in_cluster2)

                center_1 = find_centroid(data_points_cluster1)
                center_2 = find_centroid(data_points_cluster2)

                distance = find_distance(center_1, center_2)

                if(distance < min_distance):
                    min_distance = distance
                    cluster1 = key1
                    cluster2 = key2

    size_of_cluster.append(min(len(dict_of_clusters1[cluster1]), len(dict_of_clusters1[cluster2])))

    # Adds cluster 2 data records to cluster 1
    dict_of_clusters1[cluster1].extend(dict_of_clusters1[cluster2])

    # removes the cluster 2
    dict_of_clusters1.pop(cluster2, None)

    # Stores the 6 cluster records
    if (len(dict_of_clusters1) == 6):
        dict_of_six_clusters = copy.deepcopy(dict_of_clusters1)

    # Breaks when there's only one cluster
    if (len(dict_of_clusters1) == 1):
        break

    index += 1
    print("Iteration: ", index)

print("Size of smallest cluster in last 20 merges:")
print(size_of_cluster[-20:])

# Prints the cluster centers for six clusters
cluster_centers = []                            # Keeps track of cluster centers
for key, val in dict_of_six_clusters.items():
    data_point = []
    for record in val:
        data_point.append(dict_of_records[record])

    cluster_centers.append(find_centroid(data_point))
print("Cluster Centers:")
print(cluster_centers)

cluster_size = []                               # Keeps track of size of clusters
for key, val in dict_of_six_clusters.items():
    cluster_size.append(len(val))
print("Six cluster sizes:")
print(sorted(cluster_size))