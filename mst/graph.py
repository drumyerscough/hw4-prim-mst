import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        self.mst = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')
        
    def _check_adj_mat(self):
        """

        A method to check that the input adjacency matrix makes sense and is suitable for the implementation
        of Prim's algorithm below. Raises various exceptions if the input is bad.

        """
        if self.adj_mat.shape[0] != self.adj_mat.shape[1]: raise Exception('This adjacency matrix is not NxN.')
        if not np.allclose(self.adj_mat, self.adj_mat.T): raise Exception('This adjacency matrix is not symmetric.')
        if self.adj_mat.dtype != int and self.adj_mat.dtype != float: raise ValueError('This adjacency matrix contains non-numeric data.')
        if np.sum(np.isnan(self.adj_mat)): raise ValueError('This adjacency matrix contains a NaN.')
        if np.sum(np.isinf(self.adj_mat)): raise ValueError('This adjacency matrix contains an infinity.')
        if np.sum(self.adj_mat < 0): raise ValueError('This adjacency matrix contains a negative edge weight.')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # check that input adj_mat makes sense
        self._check_adj_mat()

        self.mst = np.zeros(shape=self.adj_mat.shape)
        S = set() # setof already visited nodes
        T = set() # set of 2-tuples of nodes representing edges in the MST
        pred = {} # dict that maps nodes to their predecessors in the search
        pi = {} # cost of adding a node
        pq = [] # priority queue, entries will be 3-lists containing [cost, order of addition to the queue, node idx]
        heapq.heapify(pq)

        # pick a random starting node s
        s = np.random.randint(self.adj_mat.shape[0])
        S.add(s)
        pred[s] = None
        pi[s] = [0, 0, s]
        heapq.heappush(pq, pi[s])

        # initialize all nodes v to have no predecessor and infinite cost
        counter = 1 # decides which node to proceed with if two nodes have same cost
        max = np.inf
        for v in range(self.adj_mat.shape[0]):
            if v != s:
                pred[v] = None
                pi[v] = [np.inf, 0, v]
                heapq.heappush(pq, pi[v])
                counter += 1

        # iterate thru nodes in the priority queue to construct the MST
        iter = 0
        while len(pq) > 0 and iter <= self.adj_mat.shape[0]:
            u = heapq.heappop(pq)[2]
            S.add(u)
            T.add(frozenset([pred[u], u]))
            for node, edge_wt in enumerate(self.adj_mat[u, :]):
                if node not in S:
                    if 0 < edge_wt < pi[node][0]:
                        pi[node][0] = edge_wt
                        pred[node] = u
            # sort after updating costs to maintain heap invariant
            pq.sort()
            
            # keep track of number of iterations in case graph is disconnected
            iter += 1

        for (node1, node2) in T:
            if node1 != None and node2 != None:
                self.mst[node1, node2] = 1
                self.mst[node2, node1] = 1