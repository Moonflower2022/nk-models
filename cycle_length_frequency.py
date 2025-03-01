import numpy as np
import time
from statistics import *
import matplotlib.pyplot as plt
from collections import Counter

from nk_model import iterate_until_cycle

def test_iterate_until_cycle(node_num, input_num):
    print(
        "iterate_until_cycle(node_num, input_num):", iterate_until_cycle(node_num, input_num, prints=True)
    )

def graph_cycle_frequency(node_num, input_num, simulation_num):
    start_time = time.time()
    results = [iterate_until_cycle(node_num, input_num) for _ in range(simulation_num)]
    results.sort()
    print("Running time (s):", time.time() - start_time)


    print("mean:", mean(results))
    print("median:", median(results))
    print("mode:", mode(results))

    counter = Counter(results)

    values = list(counter.keys())
    frequencies = np.array(list(counter.values())) / simulation_num

    print(values)

    plt.bar(values, frequencies, color="blue")
    plt.xlabel("Cycle Lengths")
    plt.ylabel("Frequency")
    plt.title("Frequency of Cycle Lengths")
    plt.show()

if __name__ == "__main__":
    node_num = 5
    input_num = 2
    assert input_num <= node_num
    simulation_num = 100000

    graph_cycle_frequency(node_num, input_num, simulation_num)
    
