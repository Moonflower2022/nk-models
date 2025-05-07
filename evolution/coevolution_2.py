import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nk_model import NKModel
import random
import numpy as np

class NKCModel(NKModel):
    def __init__(self, *args, c, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

        self.nodes2 = self.init_nodes()
        self.input_paths2 = self.init_input_paths()

        self.other_input_paths = [random.sample(range(self.n), self.c) for _ in range(self.n)]
        self.other_input_paths2 = [random.sample(range(self.n), self.c) for _ in range(self.n)]

        self.fitness_contributions = {}
        self.fitness_contributions2 = {}

        self.current_fitness = self.fitness_function()
        self.current_fitness2 = self.fitness_function2()

        self.fitnesses = [self.current_fitness]
        self.fitnesses2 = [self.current_fitness2]
    
    def fitness_function(self):
        total = 0
        for i in range(self.n):
            correlated_gene = np.zeros(2 * self.n) - 1
            correlated_gene[i] = self.nodes[i]
            for input_index in self.input_paths[i]:
                correlated_gene[input_index] = self.nodes[input_index]
            for input_index in self.other_input_paths[i]:
                correlated_gene[input_index + self.n] = self.nodes2[input_index]
            if not tuple(correlated_gene) in self.fitness_contributions:
                self.fitness_contributions[tuple(correlated_gene)] = random.random()
            total += self.fitness_contributions[tuple(correlated_gene)]

        return total / self.n
    
    def fitness_function2(self):
        total = 0
        for i in range(self.n):
            correlated_gene = np.zeros(2 * self.n) - 1
            correlated_gene[i] = self.nodes2[i]
            for input_index in self.input_paths2[i]:
                correlated_gene[input_index] = self.nodes2[input_index]
            for input_index in self.other_input_paths2[i]:
                correlated_gene[input_index + self.n] = self.nodes[input_index]
            if not tuple(correlated_gene) in self.fitness_contributions2:
                self.fitness_contributions2[tuple(correlated_gene)] = random.random()
            total += self.fitness_contributions2[tuple(correlated_gene)]

        return total / self.n
    
    def iterate(self):
        # evolve 1

        index = random.randint(0, self.n - 1)
        self.nodes[index] = 1 - self.nodes[index]
        new_fitness = self.fitness_function()
        if self.current_fitness < new_fitness:
            self.current_fitness = new_fitness
        else:
            self.nodes[index] = 1 - self.nodes[index]

        # evolve 2

        index = random.randint(0, self.n - 1)
        self.nodes2[index] = 1 - self.nodes2[index]
        new_fitness = self.fitness_function2()
        if self.current_fitness2 < new_fitness:
            self.current_fitness2 = new_fitness
        else:
            self.nodes2[index] = 1 - self.nodes2[index]

        self.fitnesses.append(self.current_fitness)
        self.fitnesses2.append(self.current_fitness2)

    def plot(self, title="NK Model Fitness Evolution", figsize=(10, 6), save_path=None):
        """
        Plot the fitness evolution over iterations for both landscapes.
        
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
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot both fitness trajectories
        iterations = range(len(self.fitnesses))
        ax.plot(iterations, self.fitnesses, label='Landscape 1', color='blue')
        ax.plot(iterations, self.fitnesses2, label='Landscape 2', color='red')
        
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
    
if __name__ == "__main__":
    node_num = 5
    input_num = 2
    other_input_num = 1
    nkc_model = NKCModel(n=node_num, k=input_num, c=other_input_num)

    iterations = 100
    for _ in range(iterations):
        nkc_model.iterate()

    nkc_model.plot()