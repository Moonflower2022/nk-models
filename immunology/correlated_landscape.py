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
    best_score = 0
    for i in range(model.n):
        model.nodes[i] = 1 - model.nodes[i]
        new_score = fitness_function(model)
        if new_score > best_score:
            best_score = new_score
            best_nodes = [*model.nodes]

        model.nodes[i] = 1 - model.nodes[i]
    model.nodes = best_nodes

def run_experiment(node_num, inputs_num, iterations):
    fitness_contributions = {tuple(nodes): np.random.random(node_num) for nodes in get_all_nodes(node_num)}
    def fitness_function(model):
        fitness_contribution = fitness_contributions[tuple(model.nodes)]

        return sum([fitness_contribution.dot(get_gene_correlation(model, i)) for i in range(model.n)]) / model.n

    best_indicies = []
    for _ in range(iterations):
        model = NKModel(node_num, inputs_num)
        fitnesses = []
        for nodes in get_all_nodes(node_num):
            model.nodes = nodes
            fitnesses.append(fitness_function(model))
        fitness_sorted = sorted(fitnesses, reverse=True)
        fitness_index = {fitness: index for index, fitness in enumerate(fitness_sorted)}

        model.nodes = model.init_nodes()

        best_score = 0

        while True:
            climb(model, fitness_function)
            new_score = fitness_function(model)
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
    inputs_num = 2

    model = NKModel(node_num, inputs_num)

    fitness_contributions = {tuple(nodes): np.random.random(node_num) for nodes in get_all_nodes(node_num)}
    def fitness_function(model):
        fitness_contribution = fitness_contributions[tuple(model.nodes)]

        return sum([fitness_contribution.dot(get_gene_correlation(model, i)) for i in range(model.n)]) / model.n
    
    fitnesses = []
    for nodes in get_all_nodes(node_num):
        model.nodes = nodes
        fitnesses.append(fitness_function(model))
    decimalizer = np.array([2 ** i for i in range(node_num)])

    plt.bar([decimalizer.dot(np.array(nodes)) for nodes in get_all_nodes(node_num)], fitnesses)
    plt.title(f"landscape with {node_num} nodes")
    plt.ylabel("fitness")
    plt.xlabel("genotype in decimal form")
    plt.show()

if __name__ == "__main__":
    plot_landscape()