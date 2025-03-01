import random
from itertools import product
import copy
from collections import defaultdict
import graphviz

def add_binary_array(nodes):
    """turn a list of boolean ints that represents binary (left is higher) to a decimal int"""
    binary_string = "".join(map(str, nodes))
    return int(binary_string, 2)


def generate_input_paths(node_num, input_num):
    return [random.sample(range(node_num), input_num) for _ in range(node_num)]


def generate_function(input_num):
    return [random.choice([0, 1]) for _ in range(2**input_num)]


def generate_functions(node_num, input_num):
    return [generate_function(input_num) for _ in range(node_num)]


def get_new_node(nodes, input_paths, functions, i):
    return functions[i][
        sum(nodes[input_paths[i][j]] * 2**j for j in range(len(input_paths[i])))
    ]


def iterate(nodes, input_paths, functions):
    return tuple(
        [get_new_node(nodes, input_paths, functions, i) for i in range(len(nodes))]
    )


def indicies(arr, ele):
    return [index for index, value in enumerate(arr) if value == ele]  # indicies


def iterate_until_cycle(
    node_num,
    input_num,
    prints=False,
    nodes=None,
    input_paths=None,
    functions=None,
):
    if not nodes:
        nodes = [random.choice([0, 1]) for _ in range(node_num)]
    if not input_paths:
        input_paths = generate_input_paths(node_num, input_num)
    if not functions:
        functions = generate_functions(node_num, input_num)

    iterations = 2**node_num
    nodes_history = []

    for i in range(iterations + 1):
        if prints:
            print(nodes)
            print(add_binary_array(nodes))
        if nodes_history.count(nodes) > 0:
            if prints:
                print(f"cycle of length {i - indicies(nodes_history, nodes)[0]}!")
                print(f"for {nodes}")
            return i - indicies(nodes_history, nodes)[0]
        nodes_history.append(nodes)
        nodes = iterate(nodes, input_paths, functions)
    pass


def get_num_cycles(node_num, input_num, verbose=False, input_paths=None, functions=None):
    cycle_starts = []
    cycled_nodes = []

    all_nodes = product((0, 1), repeat=node_num)
    input_paths = input_paths or generate_input_paths(node_num, input_num)
    functions = functions or generate_functions(node_num, input_num)
    for nodes in all_nodes:
        if nodes in cycled_nodes:
            continue
        current_nodes = nodes
        current_history = []
        while True:
            if current_nodes in current_history:
                if not current_nodes in cycle_starts:
                    if verbose:
                        print(current_nodes)
                        print(f"found cycle. ")
                    cycle_starts.append(current_nodes)
                break
            cycled_nodes.append(current_nodes)
            current_history.append(current_nodes)
            if verbose:
                print(current_nodes)
            current_nodes = iterate(current_nodes, input_paths, functions)
    if verbose:
        print("input paths:", input_paths)
        print("functions:", functions)

    return len(cycle_starts)

def analyze(model):
    print(model.get_nodes_attractor_pairs())
    print(model.get_num_attrators())
    print(model.get_attractor_values())
    model.generate_graph()

class NKModel:
    def __init__(self, n, k, input_paths=None, functions=None, nodes=None):
        # TODO: CURRENTLY HAS WEIRD IMPLEMENTATION OF ITERATE BECAUSE SOME SCRIPTS USE THE self.nodes attribute and some dont

        self.n = n  # node_num
        self.k = k  # input_num
        self.input_paths = input_paths or self.init_input_paths()
        self.functions = functions or self.init_functions()
        self.nodes = nodes or self.init_nodes()

    def init_nodes(self):
        return [random.choice([0, 1]) for _ in range(self.n)]

    def init_input_paths(self):
        return [random.sample(range(self.n), self.k) for _ in range(self.n)]

    def init_functions(self):
        return [generate_function(self.k) for _ in range(self.n)]

    def iterate(self, nodes=None):
        if not nodes:
            self.nodes = tuple([
                get_new_node(self.nodes, self.input_paths, self.functions, i)
                for i in range(self.n)
            ])
            return self.nodes
        else:
            return tuple([
                get_new_node(nodes, self.input_paths, self.functions, i)
                for i in range(self.n)
            ])

    def get_nodes_attractor_pairs(self, verbose=False):
        nodes_attractor_pairs = {}
        visited_nodes = set()

        all_nodes = product((0, 1), repeat=self.n)

        for nodes in all_nodes:
            self.nodes = nodes
            if nodes in visited_nodes:
                continue
                
            current_history = []

            while True:
                if self.nodes in current_history:
                    cycle_start_index = current_history.index(self.nodes)
                    attractor = current_history[
                        cycle_start_index
                    ] 

                    for past_node in current_history:
                        if past_node == (0, 0, 0, 1, 0):
                            print("iersnt", current_history)
                            print("attr", attractor)
                        nodes_attractor_pairs[past_node] = attractor
                        visited_nodes.add(past_node)

                    break

                current_history.append(self.nodes)
                visited_nodes.add(self.nodes)

                if verbose:
                    print(self.nodes)

                self.iterate()

        return nodes_attractor_pairs
    
    def get_num_attrators(self):
        return len(set(self.get_nodes_attractor_pairs().values()))
    
    def get_attractor_values(self):
        attractor_values = defaultdict(list)
        for value, attractor in self.get_nodes_attractor_pairs().items():
            attractor_values[value].append(attractor)
        return attractor_values
    
    def track_progression(self):
        seen_nodes = []
        while not self.nodes in seen_nodes:
            print(self.nodes)
            seen_nodes.append(self.nodes)
            self.iterate()
        print(self.nodes)

    def generate_graph(self):
        dot = graphviz.Digraph(comment='NKModel Graph')
        visited_nodes = set()

        all_nodes = product((0, 1), repeat=self.n)

        for nodes in all_nodes:
            if nodes in visited_nodes:
                continue
            dot.node(str(nodes))
            current_nodes = nodes
            while not current_nodes in visited_nodes:
                visited_nodes.add(current_nodes)
                next_nodes = self.iterate(current_nodes)
                dot.edge(str(current_nodes), str(next_nodes))
                current_nodes = next_nodes
        dot.view()


    def __str__(self):
        return f"<{self.__class__} object (input_paths: {self.input_paths}, functions: {self.functions})>"
    
    def __repr__(self):
        return self.__str__()


class MutableNKModel(NKModel):
    def __init__(self, *args, mutation_probability=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.mutation_probability = mutation_probability

    def copy(self):
        return MutableNKModel(self.n, self.k, mutation_probability=self.mutation_probability, input_paths=[input_path.copy() for input_path in self.input_paths], functions=[function.copy() for function in self.functions])

    def mutate_input_paths(self):
        for i in range(len(self.input_paths)):
            for j in range(len(self.input_paths[i])):
                if random.random() < self.mutation_probability:
                    self.input_paths[i][j] = random.choice(
                        list(set(range(self.n)) - {self.input_paths[i][j]})
                    )

    def mutate_functions(self):
        for i in range(len(self.functions)):
            for j in range(len(self.functions[i])):
                if random.random() < self.mutation_probability:
                    self.functions[i][j] = 1 - self.functions[i][j]

    def mutate(self):
        self.mutate_input_paths()
        self.mutate_functions()

class EvolutionaryNKModel(MutableNKModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def copy(self):
        return EvolutionaryNKModel(self.n, self.k, mutation_probability=self.mutation_probability, input_paths=[input_path.copy() for input_path in self.input_paths], functions=[function.copy() for function in self.functions])

    def robustness_score(self):
        base_model = self.copy()
        total_percentage_same_attractor = 0

        for i in range(len(self.input_paths)):
            for j in range(len(self.input_paths[i])):
                temp = self.input_paths[i][j]
                self.input_paths[i][j] = random.choice(
                    list(set(range(self.n)) - {self.input_paths[i][j]})
                )
                total_percentage_same_attractor += self.percentage_same_attractor(base_model)
                self.input_paths[i][j] = temp
                    

        for i in range(len(self.functions)):
            for j in range(len(self.functions[i])):
                self.functions[i][j] = 1 - self.functions[i][j]
                total_percentage_same_attractor += self.percentage_same_attractor(base_model)
                self.functions[i][j] = 1 - self.functions[i][j]
        return total_percentage_same_attractor / (len(self.input_paths) * len(self.input_paths[0]) + len(self.functions) * len(self.functions[0]))
    
    def low_attractor_num_penalty(self):
        if self.get_num_attrators() == 1:
            return -0.5
        return 0
    
    def fitness_score(self):
        return self.robustness_score() + self.low_attractor_num_penalty()

    def percentage_same_attractor(self, other_nk_model):
        nodes_attractor_pairs = self.get_nodes_attractor_pairs()
        other_nodes_attractor_pairs = other_nk_model.get_nodes_attractor_pairs()

        total_attractors = 0
        identical_attractors = 0

        for attractor, other_attractor in zip(
            nodes_attractor_pairs.keys(), other_nodes_attractor_pairs.keys()
        ):
            if attractor == other_attractor:
                identical_attractors += 1
            total_attractors += 1
        return identical_attractors / total_attractors