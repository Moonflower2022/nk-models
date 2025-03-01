from nk_model import EvolutionaryNKModel
from itertools import product
import time

def select(models, top_n):
    sorted_models = sorted(models, key=lambda model: model.fitness_score(), reverse=True)
    return sorted_models[:top_n]

def reproduce(models, reproduction_rate):
    new_models = []
    for model in models:
        for _ in range(reproduction_rate):
            new_model = model.copy()
            new_model.mutate()
            new_models.append(new_model)
    return new_models

def test_mutations(N, K, mutation_probability):
    model = EvolutionaryNKModel(N, K, mutation_probability=mutation_probability)
    print(model)
    model.mutate_functions()
    print(model)
    model.mutate_input_paths()
    print(model)
    print(model.fitness_score())

def evolve(N, K, mutation_probability, iterations, reproduction_rate, top_n):
    model = EvolutionaryNKModel(N, K, mutation_probability=mutation_probability)

    models = reproduce([model for _ in range(top_n)], reproduction_rate)

    for _ in range(iterations):
        models = select(models, top_n)
        print("best fitness:", models[0].fitness_score())
        with open("evolved_nk_models.txt", "w+") as output_file:
            for model in models:
                output_file.write(f"[{model.input_paths}, {model.functions}]\n")
        models = reproduce(models, reproduction_rate)

    models = select(models, top_n)

    with open("evolved_nk_models.txt", "w+") as output_file:
        for model in models:
            output_file.write(f"[{model.input_paths}, {model.functions}]\n")

if __name__ == "__main__":
    N = 10
    K = 2
    mutation_probability = 0.1

    # test_mutations(N, K, mutation_probability)

    iterations = 5
    reproduction_rate = 5
    top_n = 10

    start_time = time.time()
    evolve(N, K, mutation_probability, iterations, reproduction_rate, top_n)
    print("Training Time (s):", time.time() - start_time)