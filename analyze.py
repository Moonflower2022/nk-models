from nk_model import NKModel, analyze

if __name__ == "__main__":
    n = 5
    k = 2

    input_paths = [[0, 2], [3, 4], [0, 3], [2, 4], [0, 3]]
    functions = [[0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1]]

    model = NKModel(n, k, input_paths=input_paths, functions=functions)
    analyze(model)

    while True:
        nodes_string = input("Input a starting state with spaces between 0/1: ")
        nodes = tuple([int(char) for char in nodes_string.split(" ")])

        model.nodes = nodes
        model.track_progression()