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
    
    def iterate(self, num_mutations_to_try1=1, num_mutations_to_try2=1):
        """
        Perform one iteration of the coevolutionary process
        
        Parameters:
        -----------
        num_mutations_to_try1 : int
            Number of random mutations to try for species 1 before selecting the best one
        num_mutations_to_try2 : int
            Number of random mutations to try for species 2 before selecting the best one
        """
        # Evolve species 1
        if self.n1 > 0:  # Only evolve if there are nodes
            best_fitness1 = self.current_fitness1
            best_nodes1 = self.nodes1.copy()
            
            # Try multiple possible mutations
            for _ in range(num_mutations_to_try1):
                # Make a copy of the current state
                test_nodes1 = self.nodes1.copy()
                
                # Try a random mutation
                index = random.randint(0, self.n1 - 1)
                test_nodes1[index] = 1 - test_nodes1[index]  # Flip the bit
                
                # Temporarily apply the mutation to check fitness
                original_nodes1 = self.nodes1.copy()
                self.nodes1 = test_nodes1
                new_fitness = self.fitness_function1()
                
                # If this is the best mutation so far, remember it
                if new_fitness > best_fitness1:
                    best_fitness1 = new_fitness
                    best_nodes1 = test_nodes1.copy()
                
                # Restore original state for next trial
                self.nodes1 = original_nodes1
            
            # Apply the best mutation found (if any improved fitness)
            if best_fitness1 > self.current_fitness1:
                self.nodes1 = best_nodes1
                self.current_fitness1 = best_fitness1

        # Evolve species 2
        if self.n2 > 0:  # Only evolve if there are nodes
            best_fitness2 = self.current_fitness2
            best_nodes2 = self.nodes2.copy()
            
            # Try multiple possible mutations
            for _ in range(num_mutations_to_try2):
                # Make a copy of the current state
                test_nodes2 = self.nodes2.copy()
                
                # Try a random mutation
                index = random.randint(0, self.n2 - 1)
                test_nodes2[index] = 1 - test_nodes2[index]  # Flip the bit
                
                # Temporarily apply the mutation to check fitness
                original_nodes2 = self.nodes2.copy()
                self.nodes2 = test_nodes2
                new_fitness = self.fitness_function2()
                
                # If this is the best mutation so far, remember it
                if new_fitness > best_fitness2:
                    best_fitness2 = new_fitness
                    best_nodes2 = test_nodes2.copy()
                
                # Restore original state for next trial
                self.nodes2 = original_nodes2
            
            # Apply the best mutation found (if any improved fitness)
            if best_fitness2 > self.current_fitness2:
                self.nodes2 = best_nodes2
                self.current_fitness2 = best_fitness2

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


def run_multiple_simulations(n1, k1, c1, num_mutations_to_try1, n2, k2, c2, num_mutations_to_try2, iterations=100, num_runs=30):
    """
    Run multiple simulations with the same parameters and collect statistics
    
    Parameters:
    -----------
    n1, k1, c1 : int
        Parameters for species 1
    n2, k2, c2 : int
        Parameters for species 2
    iterations : int
        Number of iterations to run each simulation
    num_runs : int
        Number of simulations to run
    
    Returns:
    --------
    dict
        Statistics about the runs including win ratios and average fitness trajectories
    """
    # Initialize arrays to store fitness trajectories
    all_fitness1 = np.zeros((num_runs, iterations + 1))
    all_fitness2 = np.zeros((num_runs, iterations + 1))
    
    # Track wins (which species had higher final fitness)
    wins1 = 0
    wins2 = 0
    ties = 0
    
    # Run the simulations
    for run in range(num_runs):
        print(f"run {run+1}/{num_runs}", end="\r")
        # Create a new model instance
        model = NKCModelAsymmetric(n1=n1, k1=k1, c1=c1, n2=n2, k2=k2, c2=c2)
        
        # Record initial fitness
        all_fitness1[run, 0] = model.current_fitness1
        all_fitness2[run, 0] = model.current_fitness2
        
        # Run the simulation
        for iter in range(iterations):
            model.iterate(num_mutations_to_try1=num_mutations_to_try1, num_mutations_to_try2=num_mutations_to_try2)
            all_fitness1[run, iter + 1] = model.current_fitness1
            all_fitness2[run, iter + 1] = model.current_fitness2
        
        # Determine the winner
        if model.current_fitness1 > model.current_fitness2:
            wins1 += 1
        elif model.current_fitness2 > model.current_fitness1:
            wins2 += 1
        else:
            ties += 1
    
    # Calculate win ratios
    total_runs = wins1 + wins2 + ties
    win_ratio1 = wins1 / total_runs
    win_ratio2 = wins2 / total_runs
    tie_ratio = ties / total_runs
    
    # Calculate average fitness at each iteration
    avg_fitness1 = np.mean(all_fitness1, axis=0)
    avg_fitness2 = np.mean(all_fitness2, axis=0)
    
    # Calculate standard deviation of fitness at each iteration
    std_fitness1 = np.std(all_fitness1, axis=0)
    std_fitness2 = np.std(all_fitness2, axis=0)
    
    return {
        'avg_fitness1': avg_fitness1,
        'avg_fitness2': avg_fitness2,
        'std_fitness1': std_fitness1,
        'std_fitness2': std_fitness2,
        'win_ratio1': win_ratio1,
        'win_ratio2': win_ratio2,
        'tie_ratio': tie_ratio,
        'wins1': wins1,
        'wins2': wins2,
        'ties': ties,
        'total_runs': total_runs
    }

def plot_average_fitness(stats, n1, k1, c1, num_mutations_to_try1, n2, k2, c2, num_mutations_to_try2, iterations=100, figsize=(12, 8), save_path=None):
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(iterations + 1)
    
    # Plot average fitness trajectories with confidence intervals
    ax.plot(x, stats['avg_fitness1'], 'b-', label=f'Species 1 (n={n1}, k={k1}, c={c1}, mutations={num_mutations_to_try1})')
    ax.fill_between(x, 
                   stats['avg_fitness1'] - stats['std_fitness1'], 
                   stats['avg_fitness1'] + stats['std_fitness1'], 
                   color='blue', alpha=0.2)
    
    ax.plot(x, stats['avg_fitness2'], 'r-', label=f'Species 2 (n={n2}, k={k2}, c={c2}, mutations={num_mutations_to_try2})')
    ax.fill_between(x, 
                   stats['avg_fitness2'] - stats['std_fitness2'], 
                   stats['avg_fitness2'] + stats['std_fitness2'], 
                   color='red', alpha=0.2)
    
    # Add win ratio information to the plot
    win_info = (f"Win Rates (out of {stats['total_runs']} runs):\n"
                f"Species 1: {stats['win_ratio1']:.2%} ({stats['wins1']} wins)\n"
                f"Species 2: {stats['win_ratio2']:.2%} ({stats['wins2']} wins)\n"
                f"Ties: {stats['tie_ratio']:.2%} ({stats['ties']} ties)")
    
    plt.figtext(0.5, 0.01, win_info, wrap=True, horizontalalignment='center', fontsize=12)
    
    # Add labels and title
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Average Fitness')
    ax.set_title(f'Average Fitness Over {stats["total_runs"]} Simulations\n(Shaded areas show standard deviation)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for the text at the bottom
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

def run_one_match():
    # Example parameters for asymmetric species
    n1 = 5  # Number of nodes in species 1
    k1 = 2  # Number of internal connections per node in species 1
    c1 = 1  # Number of inputs from species 2 to species 1
    num_mutations_to_try1 = 1
    
    n2 = 7  # Number of nodes in species 2
    k2 = 3  # Number of internal connections per node in species 2
    c2 = 2  # Number of inputs from species 1 to species
    num_mutations_to_try2 = 1
    
    # Example 1: Run a single simulation and visualize it
    print("Running a single simulation...")
    nkc_model = NKCModelAsymmetric(n1=n1, k1=k1, n2=n2, k2=k2, c1=c1, c2=c2)
    
    iterations = 100
    for _ in range(iterations):
        nkc_model.iterate(num_mutations_to_try1=num_mutations_to_try1, num_mutations_to_try2=num_mutations_to_try2)
    
    # Plot results of single run
    nkc_model.plot_fitness(title=f"Fitness Over Time (Asymmetric NKC Model)")
    nkc_model.show_model_graph()

def run_multiple_matches():
    # Example parameters for asymmetric species
    n1 = 5  # Number of nodes in species 1
    k1 = 2  # Number of internal connections per node in species 1
    c1 = 1  # Number of inputs from species 2 to species 1
    num_mutations_to_try1 = 1
    
    n2 = 5  # Number of nodes in species 2
    k2 = 2  # Number of internal connections per node in species 2
    c2 = 1  # Number of inputs from species 1 to species 2
    num_mutations_to_try2 = 1

    iterations = 100

    # Example 2: Run multiple simulations and analyze statistics
    print("\nRunning multiple simulations to gather statistics...")
    stats = run_multiple_simulations(n1=n1, k1=k1, c1=c1, num_mutations_to_try1=num_mutations_to_try1, n2=n2, k2=k2, c2=c2, num_mutations_to_try2=num_mutations_to_try2, iterations=iterations, num_runs=10000)
    plot_average_fitness(stats, n1, k1, c1, num_mutations_to_try1, n2, k2, c2, num_mutations_to_try2, iterations)

if __name__ == "__main__":
    run_multiple_matches()
    
    