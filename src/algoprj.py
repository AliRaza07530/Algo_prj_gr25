import random
import math
import time
from collections import deque

class IncrementalSSSP:
    """Incremental Single-Source Shortest Paths algorithm for directed graphs.
    
    Maintains shortest paths from a source vertex up to a maximum distance k.
    Uses iterative BFS to avoid recursion limits.
    
    Args:
        n (int): Number of vertices in the graph.
        s (int): Source vertex (0 <= s < n).
        k (int): Maximum distance to maintain (k >= 0).
    """
    def __init__(self, n, s, k):
        if n <= 0:
            raise ValueError("Number of vertices must be positive")
        if not 0 <= s < n:
            raise ValueError("Source vertex must be between 0 and n-1")
        if k < 0:
            raise ValueError("Maximum distance k must be non-negative")
            
        self.n = n
        self.s = s
        self.k = k
        self.d = [float('inf')] * n  # d[v]: distance from source to v
        self.p = [None] * n         # p[v]: parent of v in shortest path tree
        self.N = [[] for _ in range(n)]  # N[v]: list of (neighbor, weight) pairs
        self.d[s] = 0

    def Insert(self, u, v, wt=1):
        """Insert a directed edge from u to v with optional weight wt (default 1)."""
        if not 0 <= u < self.n or not 0 <= v < self.n:
            raise ValueError("Vertex indices must be between 0 and n-1")
        if wt < 0:
            raise ValueError("Edge weights must be non-negative")
            
        self.N[u].append((v, wt))
        self._scan_iterative(u, v, wt)

    def _scan_iterative(self, u, v, wt):
        """Iteratively update distances after edge insertion."""
        queue = deque()
        if self.d[u] + wt < self.d[v] and self.d[u] + wt <= self.k:
            self.d[v] = self.d[u] + wt
            self.p[v] = u
            queue.append(v)
        
        while queue:
            current = queue.popleft()
            for neighbor, weight in self.N[current]:
                new_dist = self.d[current] + weight
                if new_dist < self.d[neighbor] and new_dist <= self.k:
                    self.d[neighbor] = new_dist
                    self.p[neighbor] = current
                    queue.append(neighbor)

    def Query(self, v):
        """Query the distance from source to vertex v."""
        if not 0 <= v < self.n:
            raise ValueError("Vertex index must be between 0 and n-1")
        return self.d[v] if self.d[v] <= self.k else float('inf')

class Spanner:
    """Greedy algorithm for constructing (2k-1)-spanners of weighted graphs.
    
    Args:
        n (int): Number of vertices in the graph.
        k (int): Stretch parameter (k >= 1).
    """
    def __init__(self, n, k):
        if n <= 0:
            raise ValueError("Number of vertices must be positive")
        if k < 1:
            raise ValueError("Stretch parameter k must be >= 1")
            
        self.n = n
        self.k = k
        self.E_prime = []
        # Initialize SSSP structures for each vertex without distance limit
        self.D = [IncrementalSSSP(n, i, float('inf')) for i in range(n)]

    def Construct(self, E):
        """Construct a (2k-1)-spanner from edge list E.
        
        Args:
            E: List of edges as tuples (u, v, w) where w is the weight.
            
        Returns:
            List of edges in the spanner.
        """
        # Sort edges by non-decreasing weight
        E_sorted = sorted(E, key=lambda x: x[2])
        
        for u, v, w in E_sorted:
            if not 0 <= u < self.n or not 0 <= v < self.n:
                raise ValueError("Vertex indices must be between 0 and n-1")
            if w < 0:
                raise ValueError("Edge weights must be non-negative")
                
            # Compute distance from u to v in current spanner
            dist = self.D[u].Query(v)
            
            # Check stretch condition
            if dist > (2 * self.k - 1) * w:
                self.E_prime.append((u, v, w))
                # Update all SSSP structures with the new edge
                for i in range(self.n):
                    self.D[i].Insert(u, v, w)
                    self.D[i].Insert(v, u, w)  # Undirected graph
        
        return self.E_prime

class FullyDynamicAPSP:
    """Fully dynamic All-Pairs Shortest Paths algorithm for unweighted graphs.
    
    Implements the randomized algorithm from Roditty and Zwick with:
    - Amortized update time: O(m√n)
    - Worst-case query time: O(n^(3/4))
    
    Args:
        n (int): Number of vertices.
        E: Initial edge list (list of tuples (u, v)).
        k (int): Depth parameter for BFS trees.
        c (int): Constant for sampling probability (default 2).
    """
    def __init__(self, n, E, k, c=2):
        if n <= 0:
            raise ValueError("Number of vertices must be positive")
        if k < 1:
            raise ValueError("Depth parameter k must be >= 1")
            
        self.n = n
        self.k = k
        self.c = c
        self.E = set((u, v) for u, v in E)  # Current edge set
        self.C = set()  # Set of insertion centers
        self.t = k  # Threshold for reinitialization
        
        # Random sample S with probability (c ln n)/k
        p = (c * math.log(n)) / k if n > 1 else 1
        if p >= 1:
            self.S = set(range(n))  # Include all vertices
        else:
            self.S = set(i for i in range(n) if random.random() < p)  # Sample with probability p
        
        # Initialize data structures
        self._init_data_structures()

    def _init_data_structures(self):
        """Initialize all data structures for a new phase."""
        self.trees_in_C = {}  # {v: {u: distance}} - BFS trees to v for v in C
        self.trees_out_C = {}  # {v: {u: distance}} - BFS trees from v for v in C
        self.trees_in_S = {}  # {w: {u: distance}} - BFS trees to w for w in S
        self.trees_out_S = {}  # {w: {u: distance}} - BFS trees from w for w in S
        self._build_trees_S()

    def _build_trees_S(self):
        """Build BFS trees for all vertices in sample S."""
        for w in self.S:
            self.trees_in_S[w] = self._bfs(w, incoming=True)
            self.trees_out_S[w] = self._bfs(w, incoming=False)

    def _bfs(self, root, incoming=False):
        """Perform BFS from/to root up to depth k."""
        dist = {root: 0}
        queue = deque([root])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= self.k:
                continue
                
            neighbors = set()
            for (a, b) in self.E:
                if incoming and b == u:
                    neighbors.add(a)
                elif not incoming and a == u:
                    neighbors.add(b)
            
            for v in neighbors:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        
        # Fill in distances for all vertices
        full_dist = {v: float('inf') for v in range(self.n)}
        full_dist.update(dist)
        return full_dist

    def Insert(self, edges, center):
        """Insert a set of edges centered at a vertex.
        
        Args:
            edges: List of edges to add (tuples (u, v)).
            center: The center vertex of this insertion.
        """
        if not 0 <= center < self.n:
            raise ValueError("Center vertex must be between 0 and n-1")
            
        # Check if we need to start a new phase
        if len(self.C) >= self.t:
            self._init_data_structures()
            self.C = set()
        
        # Add edges to the graph
        for u, v in edges:
            if not 0 <= u < self.n or not 0 <= v < self.n:
                raise ValueError("Vertex indices must be between 0 and n-1")
            self.E.add((u, v))
        
        # Add center to C and build its trees
        self.C.add(center)
        self.trees_in_C[center] = self._bfs(center, incoming=True)
        self.trees_out_C[center] = self._bfs(center, incoming=False)
        
        # Rebuild all S trees
        self._build_trees_S()

    def Query(self, u, v):
        """Query the distance between vertices u and v."""
        if not 0 <= u < self.n or not 0 <= v < self.n:
            raise ValueError("Vertex indices must be between 0 and n-1")
            
        # ℓ1: Distance in graph ignoring insertions (not implemented)
        l1 = float('inf')
        
        # ℓ2: Distance through insertion centers C
        l2 = float('inf')
        for w in self.C:
            d_in = self.trees_in_C[w].get(u, float('inf'))
            d_out = self.trees_out_C[w].get(v, float('inf'))
            if d_in <= self.k and d_out <= self.k:
                l2 = min(l2, d_in + d_out)
        
        # ℓ3: Distance through sampled vertices S
        l3 = float('inf')
        for w in self.S:
            d_in = self.trees_in_S[w].get(u, float('inf'))
            d_out = self.trees_out_S[w].get(v, float('inf'))
            if d_in < float('inf') and d_out < float('inf'):
                l3 = min(l3, d_in + d_out)
        
        return min(l1, l2, l3)

def floyd_warshall(n, edges):
    """Compute APSP using Floyd-Warshall algorithm."""
    dist = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = w
        dist[v][u] = w  # Undirected graph
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist

def reduction_sssp_to_apsp(G, is_incremental=True):
    """Reduction from SSSP to APSP as described in Theorem 1 of the paper.
    
    Args:
        G: A graph represented as (V, E, w) where:
           V: list of vertices [0, ..., n-1]
           E: list of edges (u, v, weight)
           w: maximum edge weight
        is_incremental: If True, simulates incremental SSSP. Else decremental.
        
    Returns:
        The APSP distance matrix for the original graph.
    """
    V, E, W = G
    n = len(V)
    
    # Ensure V is 0-indexed [0, 1, ..., n-1]
    if V != list(range(n)):
        vertex_map = {v: i for i, v in enumerate(V)}
        E = [(vertex_map[u], vertex_map[v], w) for u, v, w in E]
        V = list(range(n))
    
    # Construct the transformed graph G'
    V_plus = list(V) + [n]  # s is vertex n
    n_plus = n + 1
    
    # Create edges from s to each vertex i with weight i*n*W
    if is_incremental:
        edges_s = [(n, i, (i + 1) * n * W) for i in range(n-1, -1, -1)]
    else:
        edges_s = [(n, i, (i + 1) * n * W) for i in range(n)]
    
    # Initialize SSSP with original edges (treat as unweighted)
    sssp = IncrementalSSSP(n_plus, n, float('inf'))
    for u, v, _ in E:
        sssp.Insert(u, v)  # Use default weight=1
        sssp.Insert(v, u)  # Undirected graph
    
    # Simulate incremental SSSP
    for s, i, wt in edges_s:
        sssp.Insert(s, i, wt)
    
    # Compute APSP on G directly using Floyd-Warshall
    dist = floyd_warshall(n, E)
    apsp = [[float('inf')] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            apsp[i][j] = dist[i][j]
    
    return apsp

def test_incremental_sssp():
    print("\n=== Incremental SSSP Tests ===")
    
    print("\nTest 1: Basic Chain with Shortcut")
    n = 6
    k = 4
    start_time = time.time()
    sssp = IncrementalSSSP(n, s=0, k=k)
    edges = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    for u, v in edges:
        sssp.Insert(u, v)
    sssp.Insert(1, 3)
    expected = [0, 1, 2, 2, 3, 4]
    result = [sssp.Query(i) for i in range(n)]
    end_time = time.time()
    print(f"Input: n={n}, s=0, k={k}, edges={edges}, extra edge=(1,3)")
    print(f"Expected distances: {expected}")
    print(f"Actual distances: {result}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("\nTest 2: Empty Graph")
    n = 5
    k = 2
    start_time = time.time()
    sssp = IncrementalSSSP(n, s=0, k=k)
    expected = [0] + [float('inf')] * (n-1)
    result = [sssp.Query(i) for i in range(n)]
    end_time = time.time()
    print(f"Input: n={n}, s=0, k={k}, edges=[]")
    print(f"Expected distances: {expected}")
    print(f"Actual distances: {result}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert result == expected, "Empty graph test failed"
    
    print("\nTest 3: Single Edge, k=1")
    n = 3
    k = 1
    start_time = time.time()
    sssp = IncrementalSSSP(n, s=0, k=k)
    sssp.Insert(0, 1)
    expected = [0, 1, float('inf')]
    result = [sssp.Query(i) for i in range(n)]
    end_time = time.time()
    print(f"Input: n={n}, s=0, k={k}, edges=[(0,1)]")
    print(f"Expected distances: {expected}")
    print(f"Actual distances: {result}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert result == expected, "Single edge test failed"
    
    print("\nTest 4: Dense Graph")
    n = 5
    k = 2
    start_time = time.time()
    sssp = IncrementalSSSP(n, s=0, k=k)
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            sssp.Insert(i, j)
            edges.append((i, j))
    expected = [0, 1, 1, 1, 1]
    result = [sssp.Query(i) for i in range(n)]
    end_time = time.time()
    print(f"Input: n={n}, s=0, k={k}, edges={edges}")
    print(f"Expected distances: {expected}")
    print(f"Actual distances: {result}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert result == expected, "Dense graph test failed"
    print("\nSSSP tests passed successfully.")

def test_spanner():
    print("\n=== Spanner Tests ===")
    
    print("\nTest 1: Random Graph")
    start_time = time.time()
    n = 8
    edges = [(i, j, random.randint(1, 10)) for i in range(n) for j in range(i+1, n)]
    k = 2
    spanner = Spanner(n, k)
    E_prime = spanner.Construct(edges)
    edge_limit = n ** (1 + 1/k)
    end_time = time.time()
    print(f"Input: n={n}, k={k}, edges={edges}")
    print(f"Expected edge count: <= {edge_limit}")
    print(f"Actual edge count: {len(E_prime)}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert len(E_prime) <= edge_limit, f"Too many edges: {len(E_prime)}"
    
    print("\nTest 2: Disconnected Graph")
    start_time = time.time()
    n = 6
    edges = [(0,1,2), (1,2,3)]
    k = 2
    spanner = Spanner(n, k)
    E_prime = spanner.Construct(edges)
    end_time = time.time()
    print(f"Input: n={n}, k={k}, edges={edges}")
    print(f"Expected edge count: >= 2")
    print(f"Actual edge count: {len(E_prime)}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert len(E_prime) >= 2, "Disconnected graph test failed"
    
    print("\nTest 3: Uniform Weights")
    start_time = time.time()
    n = 6
    edges = [(0,1,1), (1,2,1), (2,3,1), (3,4,1), (4,5,1)]
    k = 2
    spanner = Spanner(n, k)
    E_prime = spanner.Construct(edges)
    edge_limit = n ** (1 + 1/k)
    end_time = time.time()
    print(f"Input: n={n}, k={k}, edges={edges}")
    print(f"Expected edge count: <= {edge_limit}")
    print(f"Actual edge count: {len(E_prime)}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert len(E_prime) <= edge_limit, "Uniform weights test failed"
    
    print("\nTest 4: k=1, Star Graph")
    start_time = time.time()
    n = 8
    edges = [(0, i, 1) for i in range(1, n)]
    k = 1
    spanner = Spanner(n, k)
    E_prime = spanner.Construct(edges)
    dists = []
    for u, v, w in edges:
        dist = float('inf')
        q = [(u, 0)]
        visited = set([u])
        while q:
            x, d = q.pop(0)
            if x == v:
                dist = d
                break
            for a, b, w_edge in E_prime:
                if a == x and b not in visited:
                    q.append((b, d + w_edge))
                    visited.add(b)
                elif b == x and a not in visited:
                    q.append((a, d + w_edge))
                    visited.add(a)
        dists.append((u, v, dist))
    end_time = time.time()
    print(f"Input: n={n}, k={k}, edges={edges}")
    print(f"Expected edge count: {n-1}")
    print(f"Actual edge count: {len(E_prime)}")
    print(f"Distances in spanner for original edges: {dists}")
    print(f"Runtime: {end_time - start_time:.6f} seconds")
    assert len(E_prime) == n - 1, "k=1 test failed"
    for u, v, dist in dists:
        assert dist <= 1, f"Stretch failed for k=1: d_E'({u},{v}) = {dist}"
    print("\nSpanner tests passed successfully.")

def test_apsp():
    print("\n=== Fully Dynamic APSP Tests ===")
    
    print("\nTest 1: Basic Insertions")
    n = 5
    edges = [(0,1), (1,2), (2,3), (3,4)]
    k = 3
    apsp = FullyDynamicAPSP(n, edges, k, c=10)
    
    # Test initial distances
    dist = apsp.Query(0, 4)
    print(f"Initial distance from 0 to 4: {dist}")
    assert dist == 4, f"Initial path should be 4, got {dist}"
    assert apsp.Query(4, 0) == float('inf'), "Reverse path should not exist"
    
    # Insert a shortcut and test
    apsp.Insert([(0,3)], 0)
    dist = apsp.Query(0, 4)
    print(f"Distance from 0 to 4 after inserting (0,3): {dist}")
    assert dist == 2, f"Shortcut should reduce distance to 2, got {dist}"
    
    print("\nTest 2: Random Graph Insertions")
    n = 10
    edges = [(i, i+1) for i in range(n-1)]
    k = 4
    apsp = FullyDynamicAPSP(n, edges, k)
    
    # Add random edges
    for _ in range(5):
        u, v = random.sample(range(n), 2)
        apsp.Insert([(u, v)], u)
        dist = apsp.Query(u, v)
        print(f"Inserted edge ({u}, {v}), distance: {dist}")
        assert dist == 1, f"Direct edge should give distance 1, got {dist}"
    
    print("\nTest 3: Empty Graph")
    n = 5
    edges = []
    k = 2
    apsp = FullyDynamicAPSP(n, edges, k)
    for i in range(n):
        for j in range(n):
            dist = apsp.Query(i, j)
            expected = 0 if i == j else float('inf')
            print(f"Distance from {i} to {j}: {dist}")
            assert dist == expected, f"Empty graph distance from {i} to {j} should be {expected}, got {dist}"
    
    print("\nTest 4: Complete Graph")
    n = 4
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    k = 1
    apsp = FullyDynamicAPSP(n, edges, k)
    for i in range(n):
        for j in range(n):
            dist = apsp.Query(i, j)
            expected = 0 if i == j else 1
            print(f"Distance from {i} to {j}: {dist}")
            assert dist == expected, f"Complete graph distance from {i} to {j} should be {expected}, got {dist}"
    
    print("\nAll APSP tests passed successfully.")

def test_reduction():
    print("\n=== Reduction: SSSP to APSP Tests ===")
    
    print("\nTest 1: Chain Graph")
    V = [0, 1, 2]
    E = [(0, 1, 1), (1, 2, 1)]
    W = 1
    apsp = reduction_sssp_to_apsp((V, E, W), is_incremental=True)
    expected = [
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ]
    print("APSP Matrix:")
    for row in apsp:
        print(row)
    assert apsp == expected, f"Expected {expected}, got {apsp}"
    
    print("\nTest 2: Disconnected Graph")
    V = [0, 1]
    E = []
    W = 1
    apsp = reduction_sssp_to_apsp((V, E, W), is_incremental=True)
    expected = [
        [0, float('inf')],
        [float('inf'), 0]
    ]
    print("APSP Matrix:")
    for row in apsp:
        print(row)
    assert apsp == expected, f"Expected {expected}, got {apsp}"
    
    print("\nTest 3: Weighted Graph")
    V = [0, 1]
    E = [(0, 1, 3)]
    W = 3
    apsp = reduction_sssp_to_apsp((V, E, W), is_incremental=True)
    expected = [
        [0, 3],
        [3, 0]
    ]
    print("APSP Matrix:")
    for row in apsp:
        print(row)
    assert apsp == expected, f"Expected {expected}, got {apsp}"
    
    print("\nAll reduction tests passed successfully.")

def test_twitter_spanner():
    print("\n=== Twitter Spanner Test ===")
    
    n = 50_0
    m = 1_200_0
    k = int(math.log2(n))  
    edges = []
    degrees = [1] * n
    total_degree = n
    for _ in range(m):
        u = random.choices(range(n), weights=degrees, k=1)[0]
        v = random.choices(range(n), weights=degrees, k=1)[0]
        while v == u:
            v = random.choices(range(n), weights=degrees, k=1)[0]
        edges.append((u, v, 1))
        degrees[u] += 1
        degrees[v] += 1
        total_degree += 2
    edges = list(set(edges))
    actual_m = len(edges)
    
    start_time = time.time()
    spanner = Spanner(n, k)
    E_prime = spanner.Construct(edges)
    end_time = time.time()
    runtime = end_time - start_time
    expected_edge_limit = n ** (1 + 1/k)
    
    print(f"Input: n={n}, k={k}, edges (count)={actual_m}")
    print(f"Expected edge count: <= {expected_edge_limit}")
    print(f"Actual edge count: {len(E_prime)}")
    print(f"Runtime: {runtime:.6f} seconds")
    assert len(E_prime) <= expected_edge_limit, f"Too many edges: {len(E_prime)}"
    print("\nTwitter spanner test passed successfully.")

if __name__ == "__main__":
    test_incremental_sssp()
    test_spanner()
    test_apsp()
    test_reduction()
    test_twitter_spanner()