import numpy as np

# Graph as adjacency matrix: 0 means no direct road
# Nodes: 0 - start, 4 - destination
graph = np.array([
    [0, 2, 0, 1, 0],
    [2, 0, 3, 2, 0],
    [0, 3, 0, 0, 1],
    [1, 2, 0, 0, 3],
    [0, 0, 1, 3, 0]
])

num_nodes = graph.shape[0]
start_node = 0
end_node = 4

# Parameters
num_ants = 5
num_iterations = 10
alpha = 1.0  # pheromone influence
beta = 2.0   # heuristic influence (inverse of distance)
evaporation_rate = 0.5
pheromone_deposit = 1.0

# Initialize pheromone trails with a small positive number
pheromone = np.ones((num_nodes, num_nodes)) * 0.1

# Heuristic matrix (inverse of distance, avoid division by zero)
heuristic = np.zeros_like(graph, dtype=float)
for i in range(num_nodes):
    for j in range(num_nodes):
        if graph[i, j] > 0:
            heuristic[i, j] = 1.0 / graph[i, j]

def choose_next_node(current, visited):
    probabilities = []
    neighbors = []
    for j in range(num_nodes):
        if graph[current, j] > 0 and j not in visited:
            tau = pheromone[current, j] ** alpha
            eta = heuristic[current, j] ** beta
            probabilities.append(tau * eta)
            neighbors.append(j)
    probabilities = np.array(probabilities)
    if probabilities.sum() == 0:
        return None
    probabilities = probabilities / probabilities.sum()
    return np.random.choice(neighbors, p=probabilities)

def ant_walk():
    path = [start_node]
    visited = set(path)
    current = start_node
    while current != end_node:
        next_node = choose_next_node(current, visited)
        if next_node is None:
            # Dead end: restart path (or stop)
            return None
        path.append(next_node)
        visited.add(next_node)
        current = next_node
    return path

def path_length(path):
    length = 0
    for i in range(len(path)-1):
        length += graph[path[i], path[i+1]]
    return length

best_path = None
best_length = float('inf')

for iteration in range(num_iterations):
    all_paths = []
    for _ in range(num_ants):
        path = None
        while path is None:
            path = ant_walk()
        all_paths.append(path)

    # Evaporate pheromone
    pheromone *= (1 - evaporation_rate)

    # Deposit pheromone based on path quality
    for path in all_paths:
        length = path_length(path)
        if length < best_length:
            best_length = length
            best_path = path
        deposit_amount = pheromone_deposit / length
        for i in range(len(path)-1):
            pheromone[path[i], path[i+1]] += deposit_amount
            pheromone[path[i+1], path[i]] += deposit_amount  # undirected graph

    print(f"Iteration {iteration+1}: Best path so far: {best_path} with length {best_length}")

print("\nFinal best path found:", best_path)
print("Path length:", best_length)
