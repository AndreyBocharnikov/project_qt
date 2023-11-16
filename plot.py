import matplotlib.pyplot as plt
import numpy as np


def plot(times, names, filename):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))

    # plot violin plot
    ax.violinplot(times,
                      showmeans=False,
                      showmedians=True)
    ax.set_title('Violin plot')

    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(times))],
                  labels=names)
    ax.set_xlabel('quantization')
    ax.set_ylabel('time')

    # plt.show()
    plt.savefig(filename)
