import time
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from nk_model import get_num_cycles

"""graph the number of cycles a certain assembly has, and the mean"""

if __name__ == "__main__":
    node_num = 5
    input_num = 2
    assert input_num <= node_num
    simulation_num = 1000

    start_time = time.time()
    counts = [get_num_cycles(node_num, input_num) for _ in range(simulation_num)]
    print("Running time (s):", time.time() - start_time)

    counter = Counter(counts)
        
    values = list(counter.keys())
    frequencies = np.array(list(counter.values())) / simulation_num

    plt.bar(values, frequencies, )
    plt.xlabel("Cycle Counts")
    plt.ylabel("Frequency")
    plt.title("Frequency of Cycle Counts")
    plt.show()