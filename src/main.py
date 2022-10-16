from Environment import Environment
from Agent import Agent
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = Environment()
    U1 = env.CreateUser(17.5, 10, 10, 1, False, True, True, 10, 5)
    # U2 = env.CreateUser(25, 15, 15, 1, True, True, True, 1, 5)

    agent = Agent(input_dims=[env.M1 + env.M2 + len(env.Users) + len(env.Users) * env.N + 1],
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N)

    score_history = []

    for iter in range(1):
        state = env.reset()
        score = 0
        rewards = np.zeros((1, 1000))
        sumrate = np.zeros((1, 1000))
        U1_SINR = np.zeros((1, 1000))
        # U2_SINR = np.zeros((1, 1000))

        for i in range(1000):
            action = agent.choose_action(state)
            new_state, reward, sumrate[iter][i], SINRs = env.step(action)
            agent.remember(state, action, reward, new_state)
            agent.learn()
            obs = new_state
            score += reward
            rewards[iter][i] = reward

            U1_SINR[iter][i] = SINRs[0]
            # U2_SINR[iter][i] = SINRs[1]

        score_history.append(score)
        print(
            f"{'Episode'} {iter + 1: < 4} {' | '} {'Score -> '} {score: < 10.2f} {' | '}{'Avg_Score of last 20 episodes -> '}{np.mean(score_history[-20:]): < 10.2f}")

    plt.plot(range(1, len(score_history)+1), score_history)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.grid(1)
    plt.savefig('../tmp_results/Score.png')
    plt.show()

    # rewards = np.mean(rewards, axis=0)
    plt.plot(range(1, rewards.shape[1]+1), rewards[-1, :])
    # plt.plot(range(1, len(rewards)+1), rewards)
    plt.ylabel('Rewards')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.savefig('../tmp_results/Mean_Rewards.png')
    plt.show()

    # sumrate = np.mean(sumrate, axis=0)
    plt.plot(range(1, sumrate.shape[1]+1), sumrate[-1, :])
    # plt.plot(range(1, len(sumrate)+1), sumrate)
    plt.ylabel('Sumrate')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.savefig('../tmp_results/Mean_Sumrate.png')
    plt.show()

    # U1_SINR = np.mean(U1_SINR, axis=1)
    plt.plot(range(1, U1_SINR.shape[1]+1), U1_SINR[-1, :])
    # U1_SINR = np.reshape(U1_SINR, (1, 10000))
    # plt.plot(range(1, len(U1_SINR[0])+1), U1_SINR[0])
    plt.ylabel('U1 SINR')
    plt.xlabel('Iteration')
    plt.grid(1)
    plt.savefig('../tmp_results/U1_SINR.png')
    plt.show()

    # U2_SINR = np.mean(U2_SINR, axis=1)
    # plt.plot(range(1, U2_SINR.shape[1]+1), U2_SINR[-1, :])

    # # U2_SINR = np.reshape(U2_SINR, (1, 10000))
    # # plt.plot(range(1, len(U2_SINR[0])+1), U2_SINR[0])
    # plt.ylabel('U2 SINR')
    # plt.xlabel('Iteration')
    # plt.grid(1)
    # plt.savefig('../tmp_results/U2_SINR.png')
    # plt.show()

    agent.save_models()
