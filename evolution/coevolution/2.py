import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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

    def plot_fitness(self, title="NK Model Fitness Evolution", figsize=(10, 6), save_path=None):
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
        ax.plot(iterations, self.fitnesses, label='Species 1', color='blue')
        ax.plot(iterations, self.fitnesses2, label='Species 2', color='red')
        
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
    
    def show_model_graph(self, figsize=(12, 8), node_size=500, save_path=None):
        """
        Visualize the NKC model as a graph with two NK landscapes and connections between them.
        
        Parameters:
        -----------
        figsize : tuple
            Size of the figure (width, height)
        node_size : int
            Size of the nodes in the graph
        save_path : str, optional
            If provided, save the plot to the specified path
        
        Returns:
        --------
        matplotlib.figure.Figure
            The generated figure
        """
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for both landscapes
        # First landscape nodes (blue)
        for i in range(self.n):
            G.add_node(f"A{i}", landscape="A", value=self.nodes[i])
        
        # Second landscape nodes (red)
        for i in range(self.n):
            G.add_node(f"B{i}", landscape="B", value=self.nodes2[i])
        
        # Add within-landscape edges (internal NK connections)
        for i in range(self.n):
            # First landscape internal connections
            for input_index in self.input_paths[i]:
                G.add_edge(f"A{input_index}", f"A{i}", color='blue', weight=1)
            
            # Second landscape internal connections
            for input_index in self.input_paths2[i]:
                G.add_edge(f"B{input_index}", f"B{i}", color='red', weight=1)
        
        # Add between-landscape edges (NKC connections)
        for i in range(self.n):
            # Connections from landscape B to landscape A
            for input_index in self.other_input_paths[i]:
                G.add_edge(f"B{input_index}", f"A{i}", color='purple', weight=0.5)
            
            # Connections from landscape A to landscape B
            for input_index in self.other_input_paths2[i]:
                G.add_edge(f"A{input_index}", f"B{i}", color='green', weight=0.5)
        
        # Create a layout that separates the two landscapes
        pos = {}
        
        # Position landscape A nodes on the left side
        landscape_A_pos = nx.circular_layout(nx.subgraph(G, [f"A{i}" for i in range(self.n)]))
        for node, position in landscape_A_pos.items():
            pos[node] = position - [2, 0]  # Shift to the left
        
        # Position landscape B nodes on the right side
        landscape_B_pos = nx.circular_layout(nx.subgraph(G, [f"B{i}" for i in range(self.n)]))
        for node, position in landscape_B_pos.items():
            pos[node] = position + [2, 0]  # Shift to the right
        
        # Draw nodes for landscape A (blue)
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=[f"A{i}" for i in range(self.n)],
                            node_color=['lightblue' if self.nodes[i] == 0 else 'blue' for i in range(self.n)],
                            node_size=node_size,
                            alpha=0.8,
                            ax=ax)
        
        # Draw nodes for landscape B (red)
        nx.draw_networkx_nodes(G, pos, 
                            nodelist=[f"B{i}" for i in range(self.n)],
                            node_color=['lightcoral' if self.nodes2[i] == 0 else 'red' for i in range(self.n)],
                            node_size=node_size,
                            alpha=0.8,
                            ax=ax)
        
        # Get edges by type for different colors
        blue_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'blue']
        red_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'red']
        purple_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'purple']
        green_edges = [(u, v) for u, v, d in G.edges(data=True) if d['color'] == 'green']
        
        # Draw edges with different colors and styles
        nx.draw_networkx_edges(G, pos, edgelist=blue_edges, edge_color='blue', arrows=True, width=1.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='red', arrows=True, width=1.5, ax=ax)
        nx.draw_networkx_edges(G, pos, edgelist=purple_edges, edge_color='purple', arrows=True, width=1, 
                            style='dashed', ax=ax, alpha=0.7)
        nx.draw_networkx_edges(G, pos, edgelist=green_edges, edge_color='green', arrows=True, width=1, 
                            style='dashed', ax=ax, alpha=0.7)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=2, label='Species 1 internal connections (K)'),
            plt.Line2D([0], [0], color='red', lw=2, label='Species 2 internal connections (K)'),
            plt.Line2D([0], [0], color='purple', lw=1.5, linestyle='dashed', label='Species 2 → Species 1 connections (C)'),
            plt.Line2D([0], [0], color='green', lw=1.5, linestyle='dashed', label='Species 1 → Species 2 connections (C)'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Set plot title and remove axes
        plt.title(f"NKC Model Graph (n={self.n}, k={self.k}, c={self.c})")
        plt.axis('off')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig
    
if __name__ == "__main__":
    node_num = 5
    input_num = 2
    other_input_num = 1
    nkc_model = NKCModel(n=node_num, k=input_num, c=other_input_num)

    iterations = 100
    for _ in range(iterations):
        nkc_model.iterate()

    nkc_model.plot_fitness(title=f"fitness over time (n: {node_num}, k: {input_num}, c: {other_input_num})")
    nkc_model.show_model_graph()