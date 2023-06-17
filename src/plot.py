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
    plt.plot(range(1, len(score_history) + 1),
             score_history, linewidth=1.5, label='Score')
    plt.title(title, fontweight='bold')
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(1)
    # plt.savefig('../tmp_results/Score.png')
    plt.savefig('../../Score.png', bbox_inches='tight')
    plt.show()

    # Plot the sumrate
    plt.plot(range(1, len(sumrate) + 1), sumrate,
             linewidth=1.5, label='Sumrate')
    plt.title(title, fontweight='bold')
    plt.ylabel('Sumrate')
    plt.xlabel('Iteration')
    plt.grid(1)
    moving_average = np.zeros((len(sumrate)))
    window_size = 100
    for i in range(len(sumrate)):
        if i < window_size:
            moving_average[i] = np.mean(sumrate[:i + 1])
        else:
            moving_average[i] = np.mean(sumrate[i - window_size + 1:i + 1])

    plt.plot(range(1, len(moving_average) + 1), moving_average,
             linewidth=1.3, label=f'MA 100, {moving_average[-1] : < .3}')
    # plt.axhline(y=sumrate.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=sumrate.max(), xmin=0, xmax=1,
                color='k', label=f'Max = {sumrate.max() : < .3}')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='best')
    # plt.savefig('../tmp_results/Sumrate.png')
    plt.savefig('../../Sumrate.png', bbox_inches='tight')
    plt.show()

    # Plot the U1_SINR
    plt.plot(range(1, len(u1_sinr) + 1), u1_sinr,
             linewidth=1.5, label='U1 SINR')
    # plt.yscale('log')
    plt.title(title, fontweight='bold')
    plt.ylabel('U1 SINR')
    plt.xlabel('Iteration')
    plt.grid(1)

    moving_average = np.zeros((len(u1_sinr)))
    window_size = 100
    for i in range(len(u1_sinr)):
        if i < window_size:
            moving_average[i] = np.mean(u1_sinr[:i + 1])
        else:
            moving_average[i] = np.mean(u1_sinr[i - window_size + 1:i + 1])

    plt.plot(range(1, len(moving_average) + 1), moving_average,
             linewidth=1.3, label=f'MA 100 , {moving_average[-1] : < .3}')

    # plt.axhline(y=u1_sinr.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=u1_sinr.max(), xmin=0, xmax=1, color='k',
                label=f'Max = {u1_sinr.max() : < .3}')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='best')
    # plt.savefig('../tmp_results/U1_SINR.png')
    plt.savefig('../../U1_SINR.png', bbox_inches='tight')
    plt.show()

    if u2_sinr is not None:
        # Plot the U2_SINR
        plt.plot(range(1, len(u2_sinr) + 1), u2_sinr,
                 linewidth=1.5, label='U2 SINR')
        # plt.yscale('log')
        plt.title(title, fontsize=12, fontweight='bold')
        plt.ylabel('U2 SINR')
        plt.xlabel('Iteration')
        plt.grid(1)
        moving_average = np.zeros((len(u2_sinr)))
        window_size = 100
        for i in range(len(u2_sinr)):
            if i < window_size:
                moving_average[i] = np.mean(u2_sinr[:i + 1])
            else:
                moving_average[i] = np.mean(u2_sinr[i - window_size + 1:i + 1])

        # plt.axhline(y=u2_sinr.mean(), xmin=0, xmax=1, color='r', label='Mean')
        plt.plot(range(1, len(moving_average) + 1),
                 moving_average, linewidth=1.3, label=f'MA 100 , {moving_average[-1] : < .3}')
        plt.axhline(y=u2_sinr.max(), xmin=0, xmax=1, color='k',
                    label=f'Max = {u2_sinr.max(): < .3}')

        plt.legend(bbox_to_anchor=(1.0, 1), loc='best')
        # plt.savefig('../tmp_results/U2_SINR.png')
        plt.savefig('../../U2_SINR.png', bbox_inches='tight')
        plt.show()
