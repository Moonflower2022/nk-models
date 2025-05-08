import sys
import os
import random
import numpy as np
import matplotlib.pyplot as plt

# Assuming nk_model.py is in the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nk_model import NKModel

class MultiSpeciesNKC:
    def __init__(self, n, k, c, connectivity_matrix):
        """
        Initialize a multi-species NKC model.
        
        Parameters:
        -----------
        n : int
            Number of genes for each species
        k : int
            Number of genes each gene depends on within its own species
        c : int
            Number of genes each gene depends on from each connected species
        connectivity_matrix : list of lists
            A matrix where connectivity_matrix[i][j] is True if species i is influenced by species j
        """
        self.n = n
        self.k = k
        self.c = c
        
        # Parse connectivity matrix
        self.connectivity_matrix = connectivity_matrix
        self.num_species = len(connectivity_matrix)
        
        # Initialize all species
        self.species = []
        for i in range(self.num_species):
            self.species.append({
                'nodes': self.init_nodes(),
                'input_paths': self.init_input_paths(),
                'other_input_paths': {}, # Dictionary mapping species index to list of inputs
                'fitness_contributions': {},
                'current_fitness': 0,
                'fitnesses': []
            })
            
        # Set up connections between species
        for i in range(self.num_species):
            for j in range(self.num_species):
                if i != j and self.connectivity_matrix[i][j]:
                    # Species i is influenced by species j
                    self.species[i]['other_input_paths'][j] = [
                        random.sample(range(self.n), self.c) for _ in range(self.n)
                    ]
        
        # Calculate initial fitness for all species
        for i in range(self.num_species):
            self.species[i]['current_fitness'] = self.fitness_function(i)
            self.species[i]['fitnesses'] = [self.species[i]['current_fitness']]

    def init_nodes(self):
        """Initialize binary nodes for a species."""
        return [random.randint(0, 1) for _ in range(self.n)]
    
    def init_input_paths(self):
        """Initialize input paths within a species."""
        return [random.sample([j for j in range(self.n) if j != i], self.k) for i in range(self.n)]
    
    def fitness_function(self, species_idx):
        """
        Calculate the fitness for a specific species.
        
        Parameters:
        -----------
        species_idx : int
            Index of the species to calculate fitness for
            
        Returns:
        --------
        float
            The fitness value for the species
        """
        species = self.species[species_idx]
        total = 0
        
        for i in range(self.n):
            # Calculate total size of the correlated gene vector
            # One for the current gene + inputs from own species + inputs from all connected species
            total_size = 1 + self.k
            for j in range(self.num_species):
                if j != species_idx and self.connectivity_matrix[species_idx][j]:
                    total_size += self.c
            
            # Initialize correlated gene with -1 (not used)
            correlated_gene = np.zeros(self.n * self.num_species) - 1
            
            # Set the current gene
            gene_pos = species_idx * self.n + i
            correlated_gene[gene_pos] = species['nodes'][i]
            
            # Add inputs from own species
            for input_idx in species['input_paths'][i]:
                input_pos = species_idx * self.n + input_idx
                correlated_gene[input_pos] = species['nodes'][input_idx]
            
            # Add inputs from connected species
            for j in range(self.num_species):
                if j != species_idx and self.connectivity_matrix[species_idx][j]:
                    for input_idx in species['other_input_paths'][j][i]:
                        input_pos = j * self.n + input_idx
                        correlated_gene[input_pos] = self.species[j]['nodes'][input_idx]
            
            # Check if we have a cached fitness value, otherwise generate a new one
            correlated_gene_tuple = tuple(correlated_gene)
            if correlated_gene_tuple not in species['fitness_contributions']:
                species['fitness_contributions'][correlated_gene_tuple] = random.random()
            
            total += species['fitness_contributions'][correlated_gene_tuple]
            
        return total / self.n
    
    def iterate(self):
        """Perform one iteration of the evolutionary process for all species."""
        for i in range(self.num_species):
            # Select a random gene to mutate
            index = random.randint(0, self.n - 1)
            
            # Flip the bit
            self.species[i]['nodes'][index] = 1 - self.species[i]['nodes'][index]
            
            # Calculate new fitness
            new_fitness = self.fitness_function(i)
            
            # Accept if better, otherwise revert
            if new_fitness > self.species[i]['current_fitness']:
                self.species[i]['current_fitness'] = new_fitness
            else:
                self.species[i]['nodes'][index] = 1 - self.species[i]['nodes'][index]
            
            # Record fitness
            self.species[i]['fitnesses'].append(self.species[i]['current_fitness'])
    
    def plot(self, title="Multi-Species NKC Model Fitness Evolution", figsize=(12, 8), save_path=None):
        """
        Plot the fitness evolution over iterations for all species.
        
        Parameters:
        -----------
        title : str
            Title of the plot
        figsize : tuple
            Size of the figure (width, height)
        save_path : str, optional
            If provided, save the plot to the specified path
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot fitness trajectories for each species
        iterations = range(len(self.species[0]['fitnesses']))
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_species))
        
        for i in range(self.num_species):
            ax.plot(iterations, self.species[i]['fitnesses'], 
                   label=f'Species {i+1}', 
                   color=colors[i])
        
        # Add labels and title
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title(title)
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve aesthetics
        plt.tight_layout()
        plt.show()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

# Example usage
if __name__ == "__main__":
    # Parameters
    n = 20  # Number of genes per species
    k = 2  # Number of genes each gene depends on within its species
    c = 1  # Number of genes each gene depends on from each connected species
    
    num_species = 8
    # Example with 3 species
    # connectivity_matrix[i][j] = True if species i is influenced by species j
    connectivity_matrix = [[True for _ in range(num_species)] for _ in range(num_species)]
    
    # Create model
    model = MultiSpeciesNKC(n=n, k=k, c=c, connectivity_matrix=connectivity_matrix)
    
    # Run simulation
    iterations = 500
    for _ in range(iterations):
        model.iterate()
    
    # Plot results
    model.plot(title=f"{num_species} species NKC Model Fitnesses (n: {n}, k: {k}, c: {c})")