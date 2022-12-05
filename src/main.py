from Environment import Environment
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = Environment(num_of_antennas=5, num_of_irs1=4, num_of_irs2=4,
                      path_loss_exponent=2, irs1_to_antenna=20,
                      irs2_to_antenna=20, irs1_to_irs2=10)

    U1 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=20, distance_to_irs2=10,
                        noise_var=1e-4, los_to_antenna=True, los_to_irs1=True,
                        los_to_irs2=True, sinr_threshold=10, penalty=0, allocated_power=1)

    # U2 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=10, distance_to_irs2=20,
    #                     noise_var=1e-4, los_to_antenna=False, los_to_irs1=True,
    #                     los_to_irs2=False, sinr_threshold=10, penalty=1, allocated_power=1)

    env.InitialState()

    agent = Agent(input_dims=[env.M1 + env.M2 + len(env.Users) * env.N],
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N)

    score_history = []
    rewards = np.zeros((1, 2000))
    sumrate = np.zeros((1, 2000))
    U1_SINR = np.zeros((1, 2000))
    # U2_SINR = np.zeros((1, 1000))
    Old_Avg = 0

    for ep in range(1):
        # state = env.Reset()
        state = env.state
        score = 0
        for iter in range(2000):
            action = agent.choose_action(state)

            new_state, reward, sumrate[ep][iter], SINRs = env.Step(action)

            agent.remember(state, action, reward, new_state)
            agent.learn()
            obs = new_state
            score += reward
            rewards[ep][iter] = reward

            U1_SINR[ep][iter] = SINRs[0]
            # U2_SINR[ep][iter] = SINRs[1]

        score_history.append(score)

        New_Avg = np.mean(score_history[-10:])

        if score >= Old_Avg:
            tmp_str = f"{New_Avg: < 10.2f}  +"

        else:
            tmp_str = f"{New_Avg: < 10.2f}  -"

        Old_Avg = New_Avg
        print(f"Episode{ep + 1: < 4}", f"Score -> {score: < 10.2f}",
              f"Avg-Score of last 10 episodes -> {tmp_str}")

    plt.plot(range(1, len(score_history)+1), score_history)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(1)
    plt.savefig('../tmp_results/Score.png')
    plt.show()

    # # rewards = np.mean(rewards, axis=0)
    # plt.plot(range(1, rewards.shape[1]+1), rewards[-1, :])
    # # plt.plot(range(1, len(rewards)+1), rewards)
    # plt.ylabel('Rewards')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.savefig('../tmp_results/Mean_Rewards.png')
    # plt.show()

    # sumrate = np.mean(sumrate, axis=0)
    sumrate = sumrate.reshape((1, sumrate.shape[0] * sumrate.shape[1]))
    # plt.plot(range(1, sumrate.shape[1]+1), sumrate[-1, :])
    # plt.subplot(2, 1, 1)
    plt.plot(range(1, len(sumrate[0])+1), sumrate[0], linewidth=1.3)
    plt.ylabel('Sumrate')
    plt.xlabel('Iteration')
    plt.grid(1)
    # plt.savefig('../tmp_results/Mean_Sumrate.png')
    # plt.show()

    plt.axhline(y=sumrate.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=sumrate.max(), xmin=0, xmax=1, color='k', label='Max')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.savefig('../tmp_results/Mean_Sumrate.png')
    plt.show()

    # U1_SINR = np.mean(U1_SINR, axis=1)
    # plt.plot(range(1, U1_SINR.shape[1]+1), U1_SINR[-1, :])
    U1_SINR = U1_SINR.reshape((1, U1_SINR.shape[0] * U1_SINR.shape[1]))
    # plt.subplot(2, 1, 1)
    plt.plot(range(1, len(U1_SINR[0])+1), U1_SINR[0], linewidth=1.3)
    plt.ylabel('U1 SINR')
    plt.xlabel('Iteration')
    plt.grid(1)

    plt.axhline(y=U1_SINR.mean(), xmin=0, xmax=1, color='r', label='Mean')
    plt.axhline(y=U1_SINR.max(), xmin=0, xmax=1, color='k', label='Max')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    plt.savefig('../tmp_results/U1_SINR.png')
    plt.show()

    # U2_SINR = np.mean(U2_SINR, axis=1)
    # plt.plot(range(1, U2_SINR.shape[1]+1), U2_SINR[-1, :])

    # U2_SINR = U2_SINR.reshape((1, U2_SINR.shape[0] * U2_SINR.shape[1]))
    # # plt.subplot(2, 1, 2)
    # plt.plot(range(1, len(U2_SINR[0])+1), U2_SINR[0])
    # plt.ylabel('U2 SINR')
    # plt.xlabel('Iteration')
    # plt.grid(1)

    # plt.axhline(y=U2_SINR.mean(), xmin=0, xmax=1, color='r', label='Mean')
    # plt.axhline(y=U2_SINR.max(), xmin=0, xmax=1, color='k', label='Max')
    # plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    # plt.savefig('../tmp_results/U2_SINR.png')
    # plt.show()

    # U2_SINR = np.reshape(U2_SINR, (1, 4000))
    # plt.plot(range(1, len(U2_SINR[0])+1), U2_SINR[0])
    # plt.ylabel('U2 SINR')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.savefig('../tmp_results/U2_SINR.png')
    # plt.show()

    agent.save_models()
