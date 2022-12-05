import numpy as np
import matplotlib.pyplot as plt


def plot(*, score_history, sumrate, u1_sinr, u2_sinr=None, mean: bool = False, title: str = "Title"):
    if mean:
        sumrate = np.mean(sumrate, axis=0)
        u1_sinr = np.mean(u1_sinr, axis=0)
        if u2_sinr is not None:
            u2_sinr = np.mean(u2_sinr, axis=0)
    else:
        sumrate = sumrate.reshape((sumrate.shape[0] * sumrate.shape[1], ))
        u1_sinr = u1_sinr.reshape((u1_sinr.shape[0] * u1_sinr.shape[1], ))
        if u2_sinr is not None:
            u2_sinr = u2_sinr.reshape((u2_sinr.shape[0] * u2_sinr.shape[1], ))

    # Plot the score history
    plt.plot(range(1, len(score_history) + 1), score_history)
    plt.title(title)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(1)
    plt.savefig('../tmp_results/Score.png')
    plt.show()

    # Plot the sumrate
    plt.plot(range(1, len(sumrate) + 1), sumrate, linewidth=1.3)
    plt.title(title)
    plt.ylabel('Sumrate')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.axhline(y=sumrate.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=sumrate.max(), xmin=0, xmax=1, color='k', label='Max')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.savefig('../tmp_results/Sumrate.png')
    plt.show()

    # Plot the U1_SINR
    plt.plot(range(1, len(u1_sinr) + 1), u1_sinr, linewidth=1.3)
    plt.title(title, fontsize=12, fontweight='bold')
    plt.ylabel('U1 SINR (dB)')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.axhline(y=u1_sinr.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=u1_sinr.max(), xmin=0, xmax=1, color='k', label='Max')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.savefig('../tmp_results/U1_SINR.png')
    plt.show()

    # Plot the U2_SINR
    # plt.plot(range(1, len(u2_sinr) + 1), u2_sinr, linewidth=1.3)
    # plt.title(title, fontsize=12, fontweight='bold')
    # plt.ylabel('U1 SINR (dB)')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.axhline(y=u2_sinr.mean(), xmin=0, xmax=1, color='r', label='Mean')
    # plt.axhline(y=u2_sinr.max(), xmin=0, xmax=1, color='k', label='Max')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    # plt.savefig('../tmp_results/U2_SINR.png')
    # plt.show()
