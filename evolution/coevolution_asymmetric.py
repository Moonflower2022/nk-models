import random
import numpy as np

class NKCModelAsymmetric:
    def __init__(self, n1, k1, n2, k2, c1, c2):
        """
        Initialize NKC model with different parameters for each species
        
        Parameters:
        -----------
        n1 : int
            Number of nodes in species 1
        k1 : int
            Number of inputs per node for species 1
        n2 : int
            Number of nodes in species 2
        k2 : int
            Number of inputs per node for species 2
        c1 : int
            Number of inputs from species 2 to species 1
        c2 : int
            Number of inputs from species 1 to species 2
        """
        self.n1 = n1
        self.k1 = k1
        self.c1 = c1
        
        self.n2 = n2
        self.k2 = k2
        self.c2 = c2

        # Initialize nodes for both species
        self.nodes1 = self.init_nodes(self.n1)
        self.nodes2 = self.init_nodes(self.n2)
        
        # Initialize internal input paths for both species
        self.input_paths1 = self.init_input_paths(self.n1, self.k1)
        self.input_paths2 = self.init_input_paths(self.n2, self.k2)

        # Initialize cross-species input paths
        self.cross_paths1 = [random.sample(range(self.n2), self.c1) for _ in range(self.n1)]  # Inputs from species 2 to species 1
        self.cross_paths2 = [random.sample(range(self.n1), self.c2) for _ in range(self.n2)]  # Inputs from species 1 to species 2

        # Fitness contribution lookup tables
        self.fitness_contributions1 = {}
        self.fitness_contributions2 = {}

        # Calculate initial fitness
        self.current_fitness1 = self.fitness_function1()
        self.current_fitness2 = self.fitness_function2()

        # Store fitness history
        self.fitnesses1 = [self.current_fitness1]
        self.fitnesses2 = [self.current_fitness2]
    
    def init_nodes(self, n):
        """Initialize nodes with random binary values"""
        return [random.randint(0, 1) for _ in range(n)]
    
    def init_input_paths(self, n, k):
        """Initialize random input paths for each node"""
        paths = []
        for i in range(n):
            # Create a list of indices excluding the current node
            available_indices = list(range(n))
            available_indices.remove(i)
            
            # Select k random inputs from available indices
            if k < len(available_indices):
                path = random.sample(available_indices, k)
            else:
                path = available_indices  # If k is too large, use all available indices
            
            paths.append(path)
        return paths
    
    def fitness_function1(self):
        """Calculate fitness for species 1"""
        total = 0
        for i in range(self.n1):
            # Create an array to store the state of this node and all its inputs
            # Initialize with -1 (unused positions)
            correlated_gene = np.zeros(self.n1 + self.n2) - 1
            
            # Set the current node's value
            correlated_gene[i] = self.nodes1[i]
            
            # Set internal input values
            for input_index in self.input_paths1[i]:
                correlated_gene[input_index] = self.nodes1[input_index]
            
            # Set cross-species input values
            for j, input_index in enumerate(self.cross_paths1[i]):
                correlated_gene[self.n1 + input_index] = self.nodes2[input_index]
            
            # Get or generate fitness contribution for this configuration
            key = tuple(correlated_gene)
            if key not in self.fitness_contributions1:
                self.fitness_contributions1[key] = random.random()
            
            total += self.fitness_contributions1[key]

        return total / self.n1
    
    def fitness_function2(self):
        """Calculate fitness for species 2"""
        total = 0
        for i in range(self.n2):
            # Create an array to store the state of this node and all its inputs
            # Initialize with -1 (unused positions)
            correlated_gene = np.zeros(self.n2 + self.n1) - 1
            
            # Set the current node's value
            correlated_gene[i] = self.nodes2[i]
            
            # Set internal input values
            for input_index in self.input_paths2[i]:
                correlated_gene[input_index] = self.nodes2[input_index]
            
            # Set cross-species input values
            for j, input_index in enumerate(self.cross_paths2[i]):
                correlated_gene[self.n2 + input_index] = self.nodes1[input_index]
            
            # Get or generate fitness contribution for this configuration
            key = tuple(correlated_gene)
            if key not in self.fitness_contributions2:
                self.fitness_contributions2[key] = random.random()
            
            total += self.fitness_contributions2[key]

        return total / self.n2
    
    def iterate(self):
        """Perform one iteration of the coevolutionary process"""
        # Evolve species 1
        if self.n1 > 0:  # Only evolve if there are nodes
            index = random.randint(0, self.n1 - 1)
            self.nodes1[index] = 1 - self.nodes1[index]  # Flip the bit
            new_fitness = self.fitness_function1()
            
            # Keep the change if fitness improves, revert otherwise
            if self.current_fitness1 < new_fitness:
                self.current_fitness1 = new_fitness
            else:
                self.nodes1[index] = 1 - self.nodes1[index]  # Revert

        # Evolve species 2
        if self.n2 > 0:  # Only evolve if there are nodes
            index = random.randint(0, self.n2 - 1)
            self.nodes2[index] = 1 - self.nodes2[index]  # Flip the bit
            new_fitness = self.fitness_function2()
            
            # Keep the change if fitness improves, revert otherwise
            if self.current_fitness2 < new_fitness:
                self.current_fitness2 = new_fitness
            else:
                self.nodes2[index] = 1 - self.nodes2[index]  # Revert

        # Record fitness values
        self.fitnesses1.append(self.current_fitness1)
        self.fitnesses2.append(self.current_fitness2)

    def plot_fitness(self, title="NKC Model Fitness Evolution", figsize=(10, 6), save_path=None):
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
        iterations = range(len(self.fitnesses1))
        ax.plot(iterations, self.fitnesses1, label=f'Species 1 (n={self.n1}, k={self.k1}, c={self.c1})', color='blue')
        ax.plot(iterations, self.fitnesses2, label=f'Species 2 (n={self.n2}, k={self.k2}, c={self.c2})', color='red')
        
        # Add labels and title
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fitness')
        ax.set_title(title)
        ax.legend()
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Improve aesthetics
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
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
        for i in range(self.n1):
            G.add_node(f"A{i}", landscape="A", value=self.nodes1[i])
        
        # Second landscape nodes (red)
        for i in range(self.n2):
            G.add_node(f"B{i}", landscape="B", value=self.nodes2[i])
        
        # Add within-landscape edges (internal NK connections)
        for i in range(self.n1):
            # First landscape internal connections
            for input_index in self.input_paths1[i]:
                G.add_edge(f"A{input_index}", f"A{i}", color='blue', weight=1)
        
        for i in range(self.n2):
            # Second landscape internal connections
            for input_index in self.input_paths2[i]:
                G.add_edge(f"B{input_index}", f"B{i}", color='red', weight=1)
        
        # Add between-landscape edges (NKC connections)
        for i in range(self.n1):
            # Connections from landscape B to landscape A
            for input_index in self.cross_paths1[i]:
                G.add_edge(f"B{input_index}", f"A{i}", color='purple', weight=0.5)
        
        for i in range(self.n2):
            # Connections from landscape A to landscape B
            for input_index in self.cross_paths2[i]:
                G.add_edge(f"A{input_index}", f"B{i}", color='green', weight=0.5)
        
        # Create a layout that separates the two landscapes
        pos = {}
        
        # Position landscape A nodes on the left side
        if self.n1 > 0:
            landscape_A_pos = nx.circular_layout(nx.subgraph(G, [f"A{i}" for i in range(self.n1)]))
            for node, position in landscape_A_pos.items():
                pos[node] = position - [2, 0]  # Shift to the left
        
        # Position landscape B nodes on the right side
        if self.n2 > 0:
            landscape_B_pos = nx.circular_layout(nx.subgraph(G, [f"B{i}" for i in range(self.n2)]))
            for node, position in landscape_B_pos.items():
                pos[node] = position + [2, 0]  # Shift to the right
        
        # Draw nodes for landscape A (blue)
        if self.n1 > 0:
            nx.draw_networkx_nodes(G, pos, 
                                nodelist=[f"A{i}" for i in range(self.n1)],
                                node_color=['lightblue' if self.nodes1[i] == 0 else 'blue' for i in range(self.n1)],
                                node_size=node_size,
                                alpha=0.8,
                                ax=ax)
        
        # Draw nodes for landscape B (red)
        if self.n2 > 0:
            nx.draw_networkx_nodes(G, pos, 
                                nodelist=[f"B{i}" for i in range(self.n2)],
                                node_color=['lightcoral' if self.nodes2[i] == 0 else 'red' for i in range(self.n2)],
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
            plt.Line2D([0], [0], color='blue', lw=2, label=f'Species 1 internal connections (k1={self.k1})'),
            plt.Line2D([0], [0], color='red', lw=2, label=f'Species 2 internal connections (k2={self.k2})'),
            plt.Line2D([0], [0], color='purple', lw=1.5, linestyle='dashed', label=f'Species 2 → Species 1 connections (c1={self.c1})'),
            plt.Line2D([0], [0], color='green', lw=1.5, linestyle='dashed', label=f'Species 1 → Species 2 connections (c2={self.c2})'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        
        # Set plot title and remove axes
        plt.title(f"NKC Model Graph (n1={self.n1}, k1={self.k1}, c1={self.c1} | n2={self.n2}, k2={self.k2}, c2={self.c2})")
        plt.axis('off')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig


if __name__ == "__main__":
    # Example parameters for asymmetric species
    n1 = 5  # Number of nodes in species 1
    k1 = 2  # Number of internal connections per node in species 1
    c1 = 1  # Number of inputs from species 2 to species 1
    
    n2 = 7  # Number of nodes in species 2
    k2 = 3  # Number of internal connections per node in species 2
    c2 = 2  # Number of inputs from species 1 to species 2
    
    # Create the model with asymmetric parameters
    nkc_model = NKCModelAsymmetric(n1=n1, k1=k1, n2=n2, k2=k2, c1=c1, c2=c2)

    # Run simulation
    iterations = 100
    for _ in range(iterations):
        nkc_model.iterate()

    # Plot results
    nkc_model.plot_fitness(title=f"Fitness Over Time (Asymmetric NKC Model)")