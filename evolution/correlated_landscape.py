import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nk_model import NKModel, get_all_nodes

import numpy as np
import random
import matplotlib.pyplot as plt
from collections import Counter

def get_gene_correlation(model, i):
    gene_correlation = np.zeros(model.n)
    gene_correlation[i] = model.nodes[i]
    for input_index in model.input_paths[i]:
        gene_correlation[input_index] = model.nodes[input_index]
    return gene_correlation

def climb(model, fitness_function):
    best_score = fitness_function(model)
    best_nodes = [*model.nodes]
    for i in range(model.n):
        model.nodes[i] = 1 - model.nodes[i]
        new_score = fitness_function(model)
        if new_score > best_score:
            best_score = new_score
            best_nodes = [*model.nodes]

        model.nodes[i] = 1 - model.nodes[i]
    model.nodes = best_nodes

def fitness_function(model, fitness_contributions):
    total = 0
    for i in range(model.n):
        correlated_gene = np.zeros(model.n) - 1
        correlated_gene[i] = model.nodes[i]
        for input_index in model.input_paths[i]:
            correlated_gene[input_index] = model.nodes[input_index]
        if not tuple(correlated_gene) in fitness_contributions:
            fitness_contributions[tuple(correlated_gene)] = random.random()
        total += fitness_contributions[tuple(correlated_gene)]

    return total / model.n

def run_experiment(node_num, inputs_num, iterations):
    fitness_contributions = {}

    default_fitness_function = lambda model: fitness_function(model, fitness_contributions)

    best_indicies = []
    for _ in range(iterations):
        model = NKModel(node_num, inputs_num)
        fitnesses = []
        for nodes in get_all_nodes(node_num):
            model.nodes = nodes
            fitnesses.append(default_fitness_function(model, ))
        fitness_sorted = sorted(fitnesses, reverse=True)
        fitness_index = {fitness: index for index, fitness in enumerate(fitness_sorted)}

        model.nodes = model.init_nodes()

        best_score = 0

        while True:
            climb(model, default_fitness_function)
            new_score = default_fitness_function(model)
            if new_score <= best_score:
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
    inputs_num = 2

    model = NKModel(node_num, inputs_num)

    fitness_contributions = {}
    default_fitness_function = lambda model: fitness_function(model, fitness_contributions)

    fitnesses = []
    for nodes in get_all_nodes(node_num):
        model.nodes = nodes
        fitnesses.append(default_fitness_function(model))
    decimalizer = np.array([2 ** i for i in range(node_num)])

    plt.bar([decimalizer.dot(np.array(nodes)) for nodes in get_all_nodes(node_num)], fitnesses)
    plt.title(f"landscape with {node_num} nodes")
    plt.ylabel("fitness")
    plt.xlabel("genotype in decimal form")
    plt.show()

def plot_peak_neighborhood(compact=False):
    """
    Finds a peak using the climb function and plots the fitness values of all nodes
    that are one mutation away from it. Optimized for performance.
    
    Parameters:
    -----------
    compact : bool, optional
        If True, creates a compact version of the plot without labels and with narrow bars.
        Default is False.
    """
    node_num = 100
    inputs_num = 2
    
    # Create the model
    model = NKModel(node_num, inputs_num)
    
    # Create fitness function - optimize by pre-calculating only what's needed
    # Instead of generating all possible node combinations, we'll generate fitness
    # contributions only for the nodes we evaluate
    fitness_contributions = {}
    default_fitness_function = lambda model: fitness_function(model, fitness_contributions)
    
    
    # Initialize model with random starting position and find peak
    model.nodes = model.init_nodes()
    
    # Direct hill climbing to find a peak
    current_fitness = default_fitness_function(model)
    improved = True
    
    while improved:
        improved = False
        climb(model, default_fitness_function)
        new_fitness = default_fitness_function(model)
        if new_fitness > current_fitness:
            current_fitness = new_fitness
            improved = True
        else:
            break
    
    peak_fitness = current_fitness
    print(f"Found peak with fitness: {peak_fitness:.6f}")
    
    # Store original peak nodes
    original_nodes = model.nodes.copy()
    
    # Pre-allocate arrays for faster operations
    neighbor_fitnesses = np.zeros(node_num)
    
    # Generate fitness values for all neighbors
    for i in range(node_num):
        # Mutate one bit
        model.nodes[i] = 1 - model.nodes[i]
        
        # Calculate fitness of this neighbor
        neighbor_fitnesses[i] = default_fitness_function(model)
        
        # Restore the original node
        model.nodes[i] = original_nodes[i]
    
    # Create a simple linear scale for x-axis
    positions = np.arange(node_num)
    
    # Create the figure - compact version has smaller figure size
    fig_size = (12, 6) if not compact else (8, 4)
    plt.figure(figsize=fig_size)
    
    # Create a bar plot with neighbor fitness values
    if compact:
        # For compact mode, make bars touch each other
        bar_width = 1.0
        plt.bar(positions, neighbor_fitnesses, width=bar_width, color='blue', align='edge', edgecolor=None)
    else:
        bar_width = 0.8
        plt.bar(positions, neighbor_fitnesses, width=bar_width, color='blue')
    
    # Add a horizontal line for the peak node's fitness
    plt.axhline(y=peak_fitness, color='red', linestyle='-', 
                label=None if compact else f'Peak Fitness: {peak_fitness:.6f}')
    
    title = f"Neighborhood of peak in landscape with {node_num} nodes and K={inputs_num}"
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
        plt.tick_params(axis='both', which='both', labelsize=8)
    
    plt.tight_layout()
    plt.show()

    print(f"Peak node fitness: {peak_fitness:.6f}")
    print(f"Average neighbor fitness: {np.mean(neighbor_fitnesses):.4f}")
    print(f"Max neighbor fitness: {max(neighbor_fitnesses):.4f}")
    print(f"Min neighbor fitness: {min(neighbor_fitnesses):.4f}")

if __name__ == "__main__":
    # plot_efficiency()
    # plot_landscape()
    plot_peak_neighborhood(compact=True)