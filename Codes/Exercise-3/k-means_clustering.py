import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def initials(data, k:int, blind:bool):
    if blind:
        # Select random k points as initial centers
        centers = data.sample(k).values.tolist()
        # centers = data.sample(k, random_state=1)
        print(centers)
        return centers
    else:
        # Select explicit k points from each class as initial centers
        # TODO: centers = data.sample(k)
        # centers = data.sample(k, random_state=1)
        return centers


def clustering(data, centers, k:int):
    clusters = []
    for i in range(k):
        clusters.append([])
    # Assign each data point to the closest center
    for point in data.values:
        min_distance = np.Infinity
        for i in range(k):
            distance = np.linalg.norm(point - centers[i])
            if distance <= min_distance:
                reference = i
                min_distance = distance
        clusters[reference].append(point.tolist()) # TODO: Might need fixing
    # print(clusters)
    return clusters


def recentering(clusters, k:int):
    # Calculate the mean of each cluster and update the centroid
    new_centers = []
    for i in range(k):
        new_centers.append(np.mean(clusters[i], axis=0).tolist())
    return new_centers


def kmeans(data, k, blind=True, tol=0):
    # Initialize iteration counter
    iter = 0
    # Initialize k centers
    centers = initials(data, k, blind)
    # Initialize clusters
    clusters = clustering(data, centers, k)
    # Repeat until convergence ## (or iteration limit)
    while True:
        # Calculate the mean of each cluster and update the centroid
        new_centers = recentering(clusters, k)
        # Check for convergence
        if np.allclose(centers, new_centers, atol=tol):
            print('Converged at iteration', iter)
            return new_centers, clusters       
        # Assign each data point to the closest center
        clusters = clustering(data, new_centers,  k)
        # Update the centers
        centers = new_centers
        # Update iteration counter
        iter += 1
    # # Return the centers and clusters if the algorithm does not converge    
    # print('Not converged')
    # return centers, clusters


def plot(centers, clusters):
    # Plot the clustered data points #8000FF
    # color_list= ['#F3AA20', '#FF9CFF', '#8FDFFD']
    color_list=  ['pink', 'lime',  'cyan']
    color_list2= ['red',  'olive', 'blue']
    # plt.figure(figsize=(5, 5))
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    i = 0
    for cluster in clusters:
        np_cluster = np.array(cluster)
        ax.scatter(np_cluster[:, 0], np_cluster[:, 1], np_cluster[:, 2], c=color_list[i])
        np_centers = np.array(centers)
        ax.scatter(np_centers[i, 0], np_centers[i, 1], np_centers[i, 2], c=color_list2[i], marker='x')
        i += 1
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_zlabel('Petal Length (cm)')
    ax.legend(['Cluster 1', 'Center 1', 'Cluster 2', 'Center 2', 'Cluster 3', 'Center 3'])
    plt.title('K-means Clustering on Iris Dataset')
    plt.show()



if __name__ == '__main__':

    # Load the Iris dataset
    path = 'Iris Dataset/'
    df = pd.read_csv(path + 'Iris.csv')

    # Initialize the feature data and the number of clusters
    FourD_data = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] # Features (Amount: 4)
    n_cluster = 3                                                                       # Number of clusters (k)

    # Run the k-means clustering algorithm on the 4D data with random initialization
    # Note: The first index of centers and clusters are the cluster index
    centers, clusters = kmeans(FourD_data, n_cluster, blind=True, tol=0.0001)
    print(centers)
    # print(clusters)

    # Plot the clustered data points and the centers in 3D (Sepal Length, Sepal Width, Petal Length)
    # 4th dimension would be hard to visualize
    plot(centers, clusters)
