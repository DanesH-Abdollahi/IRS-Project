from Environment import Environment
from Agent import Agent
import numpy as np
from plot import plot
from Display import disp


if __name__ == "__main__":

    env = Environment(num_of_antennas=5, num_of_irs1=5, num_of_irs2=5,
                      path_loss_exponent=2, irs1_to_antenna=20,
                      irs2_to_antenna=20, irs1_to_irs2=10)

    U1 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=10, distance_to_irs2=20,
                        noise_var=1e-4, los_to_antenna=False, los_to_irs1=True,
                        los_to_irs2=False, sinr_threshold=1, penalty=0, allocated_power=1, weight=0.7)

    U2 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=20, distance_to_irs2=10,
                        noise_var=1e-4, los_to_antenna=True, los_to_irs1=True,
                        los_to_irs2=True, sinr_threshold=1, penalty=0, allocated_power=1, weight=0.3)

    agent = Agent(num_states=env.num_of_users, bound=2, batch_size=64, max_size=100000,
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N,
                  noise=0.02, alpha=0.0002, beta=0.0004, fc1=1024, fc2=512)

    num_of_episodes = 150
    num_of_iterations = 100

    score_history = np.zeros((num_of_episodes,))
    rewards = np.zeros((num_of_episodes, num_of_iterations))
    sumrate = np.zeros((num_of_episodes, num_of_iterations))
    U1_SINR = np.zeros((num_of_episodes, num_of_iterations))

    U2_SINR = np.zeros((num_of_episodes, num_of_iterations))
    # users_sinr = np.zeros(
    #     (env.num_of_users, num_of_episodes, num_of_iterations))

    Old_Avg = 0
    obs = env.State()

    for ep in range(num_of_episodes):
        score = 0
        obs = env.State()
        for iter in range(num_of_iterations):
            action = agent.choose_action(obs)

            new_state, reward, sumrate[ep][iter], SINRs = env.Step(action)

            # if iter == 0 or iter == num_of_iterations - 1:
            #     print("****************************************************************")
            #     print("action: ", np.array(action))
            #     print("state: ", obs)
            #     print("New state: ", new_state)
            #     print("SINR: ", SINRs)
            #     print("****************************************************************")

            agent.remember(obs, action, reward, new_state)
            agent.learn()
            obs = new_state
            score += reward
            rewards[ep][iter] = reward

            U1_SINR[ep][iter] = SINRs[0]
            U2_SINR[ep][iter] = SINRs[1]

        # agent.learn()
        score = score / num_of_iterations
        score_history[ep] = score
        New_Avg = score_history[:ep + 1].mean()

        disp(episod=ep, score=score, score_history=score_history,
             New_Avg=New_Avg, Old_Avg=Old_Avg, SINRs=SINRs, sumrate=sumrate[ep][iter])

        # obs = env.Reset()
        Old_Avg = New_Avg

    plot(score_history=score_history, sumrate=sumrate,
         u1_sinr=U1_SINR, u2_sinr=U2_SINR, mean=False,
         title=f"N = {env.N}, M1 = {env.M1}, M2 = {env.M2}")

    # agent.save_models()
