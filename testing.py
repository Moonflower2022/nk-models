from nk_model import NKModel, get_num_cycles, EvolutionaryNKModel

def test_get_nodes_attractor_pairs():
    model = NKModel(N, K)
    nodes_attractor_pairs = model.get_nodes_attractor_pairs()
    print(nodes_attractor_pairs)
    assert get_num_cycles(N, K, verbose=True, input_paths=model.input_paths, functions=model.functions) == len(set(nodes_attractor_pairs.values()))

def test_evolutionary_model(N, K, mutation_probability):
    model = EvolutionaryNKModel(N, K, mutation_probability=mutation_probability)
    model_copy = model.copy()
    model.input_paths[0][0] = 4
    print(model.percentage_same_attractor(model_copy))

if __name__ == "__main__":
    N = 5
    K = 2

    mutation_probability = 0.1

    test_evolutionary_model(N, K, mutation_probability)
    