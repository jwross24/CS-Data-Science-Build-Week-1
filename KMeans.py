import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters=8, init='random', max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        '''Compute k-means clustering.

        Parameters
        ----------
        X : The data

        Returns
        -------
        self
            The fitted estimator
        '''
        n_samples = X.shape[0]

        if self.max_iter <= 0:
            raise ValueError("Number of iterations is invalid. "
                             f"max_iter={self.max_iter} must be > 0.")

        if n_samples < self.n_clusters:
            raise ValueError(
                f"n_samples={n_samples} must be >= "
                f"n_clusters={self.n_clusters}")

        self.centroids = {}

        # Initialize the centroids randomly or by first datapoints
        if self.init == 'random':
            seeds = np.random.permutation(n_samples)[:self.n_clusters]
        elif self.init == 'first':
            seeds = range(self.n_clusters)
        else:
            raise ValueError("init must be 'random' or 'first'")

        for i, seed in zip(range(self.n_clusters), seeds):
            self.centroids[i] = X[seed]

        # Begin k-means iterations
        for i in range(self.max_iter):
            self.clusters = {}
            for i in range(self.n_clusters):
                self.clusters[i] = []

            # Calculate the distance between each point and the centroids
            # Choose the nearest centroid
            for sample in X:
                cluster = self.predict(sample)
                self.clusters[cluster].append(sample)

            prev_centroids = dict(self.centroids)

            # Average the clusters and recalculate the centroids
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(
                    self.clusters[cluster], axis=0)

            # If the clusters change outside the tolerance, then our clusters
            # are suboptimal
            is_optimal = True

            for centroid in self.centroids:
                prev, curr = prev_centroids[centroid], self.centroids[centroid]

                if np.sum((curr - prev)/prev * 100.0) > self.tol:
                    is_optimal = False

            # If optimal, then break out of the loop
            if is_optimal:
                break

    def predict(self, sample):
        '''Predict the closest cluster for a sample in X.

        Parameters
        ----------
        sample : A line of data from X.

        Returns
        -------
        labels
            Index of the cluster for each sample.
        '''
        dist = [np.linalg.norm(sample - self.centroids[c])
                for c in self.centroids]
        cluster = dist.index(min(dist))
        return cluster


if __name__ == '__main__':
    df = pd.read_csv("./faithful.csv")
    df = df[['eruptions', 'waiting']]
    X = df.values

    kmeans = KMeans(2)
    kmeans.fit(X)

    # Plot the centroids and clusters
    colors = ['b', 'r', 'g', 'k', 'c']*10

    for i, cluster in kmeans.clusters.items():
        color = colors[i]
        for sample in cluster:
            plt.scatter(sample[0], sample[1], color=color, s=30)

    for centroid in kmeans.centroids.values():
        plt.scatter(centroid[0], centroid[1], color='black', s=60, marker='*')

    plt.show()
