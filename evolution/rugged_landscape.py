import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nk_model import NKModel, get_all_nodes
import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def climb(model, fitness_function):
    best_score = 0
    for i in range(model.n):
        model.nodes[i] = 1 - model.nodes[i]
        new_score = fitness_function[tuple(model.nodes)]
        if new_score > best_score:
            best_score = new_score
            best_nodes = [*model.nodes]

        model.nodes[i] = 1 - model.nodes[i]
    model.nodes = best_nodes

def run_experiment(node_num, inputs_num, iterations):
    fitness_function = {tuple(nodes): random.random() for nodes in get_all_nodes(node_num)}
    fitness_sorted = sorted(list(fitness_function.values()), reverse=True)
    fitness_index = {fitness: index for index, fitness in enumerate(fitness_sorted)}

    best_indicies = []
    for _ in range(iterations):
        model = NKModel(node_num, inputs_num)
        best_score = 0

        while True:
            climb(model, fitness_function)
            new_score = fitness_function[tuple(model.nodes)]
            if new_score < best_score:
                break
            best_score = new_score
        best_indicies.append(fitness_index[best_score])
    return best_indicies

def plot_efficiency():
    node_num = 5
    inputs_num = 2

    iterations = 100
    num_experiments = 1000
    indicies = []
    for _ in range(num_experiments):
        indicies += run_experiment(node_num, inputs_num, iterations)

    counter = Counter(indicies)
    print(counter)
    plt.title(f"ranking out of {2 ** node_num} from {num_experiments} experiments with {iterations} iterations each (lower is better)")
    plt.bar(counter.keys(), counter.values())
    plt.show()

def plot_landscape():
    node_num = 10

    fitness_function = {tuple(nodes): random.random() for nodes in get_all_nodes(node_num)}
    decimalizer = np.array([2 ** i for i in range(node_num)])
    plt.bar([decimalizer.dot(np.array(nodes)) for nodes in get_all_nodes(node_num)], fitness_function.values())
    plt.title(f"landscape with {node_num} nodes")
    plt.ylabel("fitness")
    plt.xlabel("genotype in decimal form")
    plt.show()


def plot_peak_neighborhood(compact=False):
    """
    Creates a random node as the peak, samples its fitness from PDF n*x^(n-1) where n = 2^node_num,
    then plots the fitness values of all nodes that are one mutation away from it.
    The fitness of each neighboring node is generated randomly.
    
    Parameters:
    -----------
    compact : bool, optional
        If True, creates a compact version of the plot without labels and with narrow bars.
        Default is False.
    """
    node_num = 100
    
    # Create a random node to serve as our peak
    peak_node = tuple(random.choice([0, 1]) for _ in range(node_num))
    
    # Set n for the PDF as 2^node_num
    n = 2**node_num
    print(f"Using n = 2^{node_num} = {n} for the PDF")
    
    # Sample the peak fitness value from [0,1] according to the PDF n*x^(n-1)
    # Using inverse transform sampling: F^(-1)(u) = u^(1/n)
    u = random.random()
    peak_fitness = u**(1/n)
    print(f"Peak fitness sampled from PDF: {peak_fitness:.6f}")
    
    # Generate fitness values for all neighbors
    neighbors = []
    neighbor_fitnesses = []
    
    for i in range(node_num):
        mutated_node = list(peak_node)
        mutated_node[i] = 1 - mutated_node[i]  # Flip one bit
        mutated_node = tuple(mutated_node)
        neighbors.append(mutated_node)
        # Generate random fitness for neighbors
        neighbor_fitnesses.append(random.random())
    
    # Create a simple linear scale for x-axis
    positions = np.arange(node_num)
    
    # Create the figure - compact version has smaller figure size
    fig_size = (12, 6) if not compact else (8, 4)
    plt.figure(figsize=fig_size)
    
    # Create a bar plot with only neighbor fitness values
    # Compact version uses bars with no spacing between them
    if compact:
        # For compact mode, make bars touch each other by setting width=1.0
        bar_width = 1.0
        # Use align='edge' to make bars flush against each other
        bars = plt.bar(positions, neighbor_fitnesses, width=bar_width, color='blue', align='edge', edgecolor=None)
    else:
        bar_width = 0.8
        bars = plt.bar(positions, neighbor_fitnesses, width=bar_width, color='blue')
    
    # Add a horizontal line for the peak node's fitness
    plt.axhline(y=peak_fitness, color='red', linestyle='-', 
                label=None if compact else f'Peak Fitness: {peak_fitness:.6f}')
    
    title = f"Neighborhood of peak in landscape with {node_num} nodes"
    if compact:
        title = f"Compact: {title}"
    plt.title(title)
    
    if not compact:
        plt.ylabel("Fitness")
        plt.xlabel("Bit position changed")
        labels = [f"Flip bit {i}" for i in range(node_num)]
        plt.xticks(positions, labels, rotation=45, ha='right')
        plt.legend()
        
        # Add text annotation for the fitness values
        for i, fitness in enumerate(neighbor_fitnesses):
            plt.text(positions[i], fitness, f"{fitness:.3f}", ha='center', va='bottom')
    else:
        # For compact version, minimal or no labels
        plt.xticks([])  # Remove x-axis ticks
        plt.tick_params(axis='both', which='both', labelsize=8)  # Smaller ticks if shown
    
    plt.tight_layout()
    plt.show()

    print(f"Peak node fitness: {peak_fitness:.6f}")
    print(f"Average neighbor fitness: {np.mean(neighbor_fitnesses):.4f}")
    print(f"Max neighbor fitness: {max(neighbor_fitnesses):.4f}")
    print(f"Min neighbor fitness: {min(neighbor_fitnesses):.4f}")

if __name__ == "__main__":
    plot_peak_neighborhood(compact=True)