import sys
import math
import time
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans

import networkx as nx

logger = logging.getLogger(__name__)

class FairletKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters, random_state, t_prime, thresh, protected_class):
        self.MCF = nx.DiGraph()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._protected_class = protected_class
        self.t_prime = t_prime
        self.thresh = thresh
        self.X = None
        self._red_blue_distances = None

    def fit(self, X, **kwargs):
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
        
        # Now that we have blue/red indices, we can perform
        # the fairlet decomp and create a new "dataset"
        # of fair centroids
        Fairlets, centroids, fairlet_costs, fairlets_to_centroids = self._MCF_fairlet_decomposition(self.X_numpy[blues,:],self.X_numpy[reds,:], blues, reds, self.X_numpy)
        
        # Now, according to Schmidt et. al 2019, we run k-means++ on the centroids
        # to compute the optimal centers, and return the resulting centers
        # Note that "changing the center of a cluster to its centroid 
        # does not violate the fairness constraint, and it is still 
        # the optimal choice for the cluster."
        # TODO: We need to 
        internal_kmeans_plus_plus = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, init='k-means++').fit(centroids)
        self.cluster_centers_ = internal_kmeans_plus_plus.cluster_centers_
        self.labels_ = internal_kmeans_plus_plus.labels_
        self.inertia_ = internal_kmeans_plus_plus.inertia_
        self.n_iter_ = internal_kmeans_plus_plus.n_iter_
        return internal_kmeans_plus_plus
        
    def predict(self, X, random_state=None):
        pass
    
    def _distance(self, x, y, axis=None):
        if isinstance(x, np.ndarray):
            assert x.shape[0] == y.shape[0], "Mismatch vector length."
        else:
            assert len(x) == len(y), "Mismatch vectors."
            x = np.array(x)
            y = np.array(y)
        if axis:
            return np.linalg.norm(x-y, axis=axis)
        return np.linalg.norm(x-y)
    
    def _construct_red_blue_distances(self, reds, blues):
        self._red_blue_distances = [[self._distance(blue, red) for red in reds] for blue in blues]
    
    def _check_balanced(self, b, r, red_count, blue_count):
        if r==0 and b==0:
            return True
        elif r==0 or b==0:
            return False 
        else:
            b1 = red_count / blue_count
            b2 = blue_count / red_count
            br = b / r
            return br <= min(b1,b2)
    
    def _plot_MCF(self):
        nx.draw_shell(self.MCF, with_labels = True) 
    
    def _MCF_fairlet_decomposition(self, blues, reds, blues_idx, reds_idx, X):
        """
        Construct our MCF graph according to Chierichetti 2017 (section 4.2)
        
        Notes:
        In networkx construction, supply is indicated with a negative demand
        weight defaults to 0 if not specified
        capacity defaults to infinity if not specified
        demand defaults to 0 if not specified
        """
        # Will contain dictionary of fairlets
        fairlet_intermediate = {}
        Fairlets = []
        
        # Construct our matrix of distances between each red and blue point
        self._construct_red_blue_distances(reds, blues)
        
        # Add the beta->rho edge, beta supply = |B|, rho demand = |R|
        # Edge capacity is min(|R|,|B|), cost is 0
        self.MCF.add_node('rho', demand = len(blues))
        self.MCF.add_node('beta', demand = (len(reds) * -1))
        self.MCF.add_edge('beta', 'rho', capacity = min(len(blues),len(reds)))
        
        # Add the reds -> rho nodes+edges
        # Then add the intermediate nodes+edges (sub_node -> red_node -> rho)
        for i in range(len(reds)):
            name = 'R'+str(i)
            self.MCF.add_node(name, demand = 1)
            self.MCF.add_edge(name, 'rho', capacity = self.t_prime - 1)
            for j in range(self.t_prime):
                sub_node_name = name + "_" + str(j)
                self.MCF.add_node(sub_node_name)
                self.MCF.add_edge(sub_node_name, name, capacity = 1)
            
        # Add the beta -> blue nodes+edges 
        # Then add the intermediate nodes+edges (beta -> blue_node -> sub_node)
        for i in range(len(blues)):
            name = 'B'+str(i)
            self.MCF.add_node(name, demand = -1)
            self.MCF.add_edge('beta', name, capacity = self.t_prime - 1)
            for j in range(self.t_prime):
                sub_node_name = name + "_" + str(j)
                self.MCF.add_node(sub_node_name)
                self.MCF.add_edge(name, sub_node_name, capacity = 1)
                
        # Connect across the bipartite nodes
        # (each sub_node -> sub_node with weight = distance!)
        for i_0 in range(len(blues)):
            for j_0 in range(self.t_prime):
                for i_1 in range(len(reds)):
                    for j_1 in range(self.t_prime):
                        sub_blue = 'B'+str(i_0)+'_'+str(j_0)
                        sub_red = 'R'+str(i_1)+'_'+str(j_1)
                        dist = self._red_blue_distances[i_0][i_1]
                        if self.thresh < dist:
                            self.MCF.add_edge(sub_blue, sub_red, weight = math.inf, capacity = 1, dist = dist)
                        else:
                            self.MCF.add_edge(sub_blue, sub_red, weight = 1, capacity = 1, dist = dist)
                            
        st = time.time()
        flow_dict = nx.min_cost_flow(self.MCF)
        cost = nx.cost_of_flow(self.MCF, flow_dict)
        print("Solution found in " + str(time.time() - st))
        
        # Loop over the flow and construct fairlets
        for key in flow_dict.keys():
            # The flow is from blue -> red, so we start with blue only
            # intermediate nodes, and look for flow
            if re.match('B\d+_\d+', key):
                if sum(flow_dict[key].values()):
                    # If we find flow, we want to create a fairlet
                    for j in flow_dict[key].keys():
                        if flow_dict[key][j] == 1:
                            # For flow we find through the intermediate red nodes,
                            # we add associated blue nodes to the fairlet
                            # rx -> red node, acting as key for fairlet
                            # bx -> blue node to add to fairlet
                            rx = j.split('_')[0]
                            bx = key.split('_')[0]
                            if reds_idx[int(rx[1:])] not in fairlet_intermediate:
                                fairlet_intermediate[reds_idx[int(rx[1:])]] = [int(bx[1:])]
                            else:
                                fairlet_intermediate[reds_idx[int(rx[1:])]].append(int(bx[1:]))
                
        Fairlets = [[key] + values for key, values in fairlet_intermediate.items()]
        
        Costs = []
        # Track the fairlet costs (max distance between points in a fairlet)
        for fairlet in Fairlets:
            costs = [max([self._distance(X[label_1], X[label_2]) for label_2 in fairlet]) for label_1 in fairlet]
            costs = sorted(costs, key=lambda c:c, reverse=False)
            cost = costs[0]
            Costs.append(cost)

        print("Number of fairlets: " + str(len(Costs)))
        
        # Create a list of centroids, which
        # will stand in for old dataset during k-means
        centroids = []
        fairlets_to_centroids = {}
        for f_ind, fairlet in enumerate(Fairlets):
            fairlet_data = np.array(X)[fairlet]
            centroid = fairlet_data.mean(axis=0)
            centroids.append(centroid)
            fairlets_to_centroids[f_ind] = centroid
        
        return Fairlets, centroids, Costs, fairlets_to_centroids