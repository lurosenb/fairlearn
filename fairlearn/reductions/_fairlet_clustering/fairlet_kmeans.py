import logging
import sys
import math
import time
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

import networkx as nx

logger = logging.getLogger(__name__)

class FairletKMeans(BaseEstimator, ClusterMixin):
    """
    An unsupervised clustering algorithm which implements the fairlet approach
    to cluster balancing.

    The fairlet decomposition approach is described in detail by
    Chierichetti, Flavio, et al. "Fair Clustering Through Fairlets."
    https://arxiv.org/abs/1802.05733

    Parameters
    ----------
    n_clusters : int
        Traditional number of clusters in unsupervised clustering problem
    random_state : int
        A seed for replicability
    t_prime : int
        From Chierichetti, Flavio, et al., t is our balance factor, where t <= 1.
        By definition t = (1 / t_prime), where t_prime > 1. So if t_prime=2, we
        are looking for a minimum balance per binary protected class of 1/2.
    thresh : float
        For larger datasets with bigger spread, this threshholds the distance
        away points can be from each other when connecting the minimum cost flow
        graph. Allows light control of the point spread.
    protected_class : string
        The binary class column name, in X, that needs protecting.
    """
    def __init__(self, n_clusters, random_state, t_prime, thresh, protected_class):
        self.MCF = nx.DiGraph()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._protected_class = protected_class
        self.t_prime = t_prime
        self.thresh = thresh
        self.X = None
        self._red_blue_distances = None
        self.internal_kmeans_plus_plus = None

    def fit(self, X, **kwargs):
        """
        Fits a balanced KMeans model by preprocessing data into fairlets
        and running traditional KMeans on their centroids, then labeling the
        data points accordingly.

        Parameters
        ----------
        X : (only) pandas.DataFrame for now
            Feature data
        """
        if isinstance(X, pd.DataFrame):
            self.pandas = True
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='ignore')
            self.X = X
            self.X_numpy = X.to_numpy()
            self.pd_cols = X.columns
            self.pd_index = X.index
        else:
            raise('Only pandas dataframes for data as of now.')
            
        reds_blues = self.X.loc[:, self.X.columns == self._protected_class]
        r_b_array = reds_blues.to_numpy()
        r_b_label = np.unique(r_b_array)
        assert len(r_b_label) == 2, 'Protected class has ' + str(len(r_b_label)) + ' options. Must be binary.'
        
        # Retrieve red/blue. Note we determine blue by constraint
        # blue >= red
        cand_1 = np.where(r_b_array == r_b_label[0])[0]
        cand_2 = np.where(r_b_array == r_b_label[1])[0]
        
        if len(cand_1) >= len(cand_2):
            blues = cand_1
            reds = cand_2
        else:
            blues = cand_2
            reds = cand_1

        # We need to make sure that there are fewer
        # reds than blues, and that reds exist
        assert len(reds) > 0, "Must have examples of both categories in protected class for the problem to be feasible."
        
        # Now that we have blue/red indices, we can perform
        # the fairlet decomp and create a new "dataset"
        # of fair centroids
        Fairlets, centroids, fairlet_costs, fairlets_to_centroids = self._MCF_fairlet_decomposition(self.X_numpy[blues,:],self.X_numpy[reds,:], blues, reds, self.X_numpy)

        # Now, according to Schmidt et. al 2019, we run k-means++ on the centroids
        # to compute the optimal centers, and return the resulting centers
        # Note that "changing the center of a cluster to its centroid 
        # does not violate the fairness constraint, and it is still 
        # the optimal choice for the cluster."
        self.internal_kmeans_plus_plus = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit(centroids)
        self.cluster_centers_ = self.internal_kmeans_plus_plus.cluster_centers_
        
        # Loop through the centroid labels, and reassign them to the original dataset points
        indx_to_centroid_label = {}
        for i, centroid_label in enumerate(self.internal_kmeans_plus_plus.labels_):
            for original_d_point in Fairlets[i]:
                indx_to_centroid_label[original_d_point] = centroid_label

        # Flatten into labels list
        X_labels = [None] * len(self.X_numpy)
        for k, v in indx_to_centroid_label.items():
            X_labels[k] = v

        self.labels_ = np.array(X_labels)
        self.inertia_ = self.internal_kmeans_plus_plus.inertia_
        self.fairlet_cost_ = sum(fairlet_costs)
        self.max_fairlet_cost_ = max(fairlet_costs)
        self.n_iter_ = self.internal_kmeans_plus_plus.n_iter_
        return self
        
    def predict(self, X, random_state=None):
        """
        Calls predict on X from the internal KMeans model, which
        is fitted to the fairlet centroids.

        NOTE: Potentially should consider balance when providing
        a cluster label. (For example, if a prediction unbalances a cluster,
        should we check the next nearest cluster?)

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Feature data
        random_state : int or RandomState instance, default=None
            Controls random numbers used for randomized predictions. Pass an
            int for reproducible output across multiple function calls.
        Returns
        -------
        Scalar or vector
            The cluster label. If `X` represents the data for a single example
            the result will be a scalar. Otherwise the result will be a vector
        """
        return self.internal_kmeans_plus_plus.n_iter_.predict(X, random_state=random_state)
    
    def _distance(self, x, y, axis=None):
        """
        Euclidean distance between two vectors,
        (order=2). The two vectors must be same length

        Parameters
        ----------
        x : numpy.ndarray or python list
        y : numpy.ndarray or python list
        Returns
        -------
        float
            Euclidean distance between the two vectors. (l2-norm)
        """
        if isinstance(x, np.ndarray):
            assert x.shape[0] == y.shape[0], "Mismatch vector length."
        else:
            assert len(x) == len(y), "Mismatch vectors."
            x = np.array(x)
            y = np.array(y)
        if axis:
            return np.linalg.norm(x-y, axis=axis, ord=2)
        return np.linalg.norm(x-y, ord=2)
    
    def _construct_red_blue_distances(self, reds, blues):
        """
        Betwene reds/blues of the protected class, calculates
        the distance between each point.

        (Sets the _red_blue_distances matrix.)

        Parameters
        ----------
        reds : numpy.ndarray or python list
        blues : numpy.ndarray or python list
        """
        self._red_blue_distances = [[self._distance(blue, red) for red in reds] for blue in blues]
    
    def _plot_MCF(self):
        """
        Draws the MCF networkx graph, for analysis.
        """
        nx.draw_shell(self.MCF, with_labels = True) 

    def _MCF_fairlet_decomposition(self, blues, reds, blues_idx, reds_idx, X):
        """
        Construct our MCF graph according to Chierichetti 2017 (section 4.2)
        
        Notes:
        In networkx construction, supply is indicated with a negative demand
        weight defaults to 0 if not specified
        demand defaults to 0 if not specified
        
        Parameters
        ----------
        blues : numpy.ndarray or python list
            Samples with majority attribute of the binary protected class 
        reds : numpy.ndarray or python list
            Samples with minority attribute of the binary protected class 
        blues_idx : numpy.ndarray or python list
            indices of above
        reds_idx : numpy.ndarray or python list
            indices of above
        X : numpy.ndarray 
            Dataset (in full)
        """
        # Time this process
        st = time.time()

        len_blues = len(blues)
        len_reds = len(reds)

        # Initializing the Graph
        self._construct_red_blue_distances(reds, blues)

        # The return structure initializations
        Fairlets = {}
        Costs = []
        Centroids = []
        Fairlets_to_Centroids = {}

        # Add the beta->rho edge, beta supply = |B|, rho demand = |R|
        # Edge capacity is min(|R|,|B|), cost is 0
        self.MCF.add_node('beta', demand=(-1*len_reds))
        self.MCF.add_node('rho', demand=(len_blues))
        self.MCF.add_edge('beta', 'rho', capacity=min(len_blues, len_reds))

        # Add the reds -> rho nodes+edges
        # Then add the intermediate nodes+edges (sub_node -> red_node -> rho)
        for i in range(len_reds):
            name = 'R'+str(i)
            self.MCF.add_node(name, demand=1)
            self.MCF.add_edge(name, 'rho', capacity=self.t_prime-1)
            for j in range(self.t_prime):
                sub_node_name = name + "_" + str(j)
                self.MCF.add_node(sub_node_name)
                self.MCF.add_edge(sub_node_name, name, capacity=1)

        # Add the beta -> blue nodes+edges 
        # Then add the intermediate nodes+edges (beta -> blue_node -> sub_node)
        for i in range(len_blues):
            name = 'B'+str(i)
            self.MCF.add_node(name,  demand=-1)
            self.MCF.add_edge('beta', name, capacity=self.t_prime-1)
            for j in range(self.t_prime):
                sub_node_name = name + '_' + str(j)
                self.MCF.add_node(sub_node_name)
                self.MCF.add_edge(name, sub_node_name, capacity=1)
                
        # Connect across the bipartite nodes
        # (each sub_node -> sub_node with weight = distance!)
        for i_0 in range(len_blues):
            for j_0 in range(self.t_prime):
                for i_1 in range(len_reds):
                    for j_1 in range(self.t_prime):
                        sub_blue = 'B'+str(i_0)+'_'+str(j_0)
                        sub_red = 'R'+str(i_1)+'_'+str(j_1)
                        dist = self._red_blue_distances[i_0][i_1]
                        if self.thresh < dist:
                            self.MCF.add_edge(sub_blue, sub_red, weight = 9999999, capacity = 1, dist = dist)
                        else:
                            self.MCF.add_edge(sub_blue, sub_red, weight = 1, capacity = 1, dist = dist)

        # Solve the mcf problem with networkX
        # TODO: Could be existent solvers that are faster        
        mcf_dict = nx.min_cost_flow(self.MCF)

        # Loop over the flow and construct Fairlets
        for key in mcf_dict.keys():
            # The flow is from blue -> red, so we start with blue only
            # intermediate nodes, and look for flow
            if re.match('B\d+_\d+', key):
                if sum(mcf_dict[key].values()) == 1:
                    # If we find flow, we want to create a fairlet
                    for j in mcf_dict[key].keys():
                        if mcf_dict[key][j] == 1:
                            # For flow we find through the intermediate red nodes,
                            # we add associated blue nodes to the fairlet
                            # rx -> red node, acting as key for fairlet
                            # bx -> blue node to add to fairlet
                            rx = j.split('_')[0]
                            bx = key.split('_')[0]
                            if reds_idx[int(rx[1:])] not in Fairlets:
                                Fairlets[reds_idx[int(rx[1:])]] = [blues_idx[int(bx[1:])]]
                            else:
                                Fairlets[reds_idx[int(rx[1:])]].append(blues_idx[int(bx[1:])])
                
        Fairlets = [[key] + values for key, values in Fairlets.items()]

        # Track the fairlet costs (minimum max distance between points in a fairlet)
        for fairlet in Fairlets:
            costs = [max([self._distance(X[label_1], X[label_2]) for label_2 in fairlet]) for label_1 in fairlet]
            Costs.append(min(costs))

        # Create a list of centroids, which
        # will stand in for old dataset during k-means
        for f_ind, fairlet in enumerate(Fairlets):
            fairlet_data = np.array(X)[fairlet]
            centroid = fairlet_data.mean(axis=0)
            Centroids.append(centroid)
            Fairlets_to_Centroids[f_ind] = centroid

        print("Fairlets and corresponding Centroids identified in " + str(time.time() - st))
        
        return Fairlets, Centroids, Costs, Fairlets_to_Centroids