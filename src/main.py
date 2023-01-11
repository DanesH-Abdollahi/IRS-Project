from Environment import Environment
from Agent import Agent
import numpy as np
from plot import plot
from Display import disp


def run(*, num_of_antennas: int = 10, num_of_irs1: int = 8, num_of_irs2: int = 8,
        path_loss_exponent: float = 2, irs1_to_antenna: float = 20, irs2_to_antenna: float = 20,
        irs1_to_irs2: float = 10, num_of_users: int = 1, distance_to_antenna: float = 40,
        distance_to_irs1: float = 20, distance_to_irs2: float = 10, noise_var: float = 1e-4,
        los_to_antenna: bool = True, los_to_irs1: bool = True, los_to_irs2: bool = True,
        sinr_threshold: float = 0, penalty: float = 9, allocated_power: float = 1,
        distance_to_antenna2: float = 0,
        distance_to_irs12: float = 0, distance_to_irs22: float = 0, noise_var2: float = 0,
        los_to_antenna2: bool = 0, los_to_irs12: bool = 0, los_to_irs22: bool = 0,
        sinr_threshold2: float = 0, penalty2: float = 0, allocated_power2: float = 0,
        num_of_iterations: int = 200, num_of_episodes: int = 1, mean_reward: bool = False):

    env = Environment(num_of_antennas=num_of_antennas, num_of_irs1=num_of_irs1,
                      num_of_irs2=num_of_irs2, path_loss_exponent=path_loss_exponent,
                      irs1_to_antenna=irs1_to_antenna, irs2_to_antenna=irs2_to_antenna,
                      irs1_to_irs2=irs1_to_irs2)

    U1 = env.CreateUser(distance_to_antenna=distance_to_antenna, distance_to_irs1=distance_to_irs1,
                        distance_to_irs2=distance_to_irs2, noise_var=noise_var,
                        los_to_antenna=los_to_antenna, los_to_irs1=los_to_irs1,
                        los_to_irs2=los_to_irs2, sinr_threshold=sinr_threshold,
                        penalty=penalty, allocated_power=allocated_power)

    # U2 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=10, distance_to_irs2=20,
    #                     noise_var=1e-4, los_to_antenna=True, los_to_irs1=True,
    #                     los_to_irs2=True, sinr_threshold=0, penalty=9, allocated_power=1)

    # env.InitialState()

    agent = Agent(input_dims=[env.num_of_users + 1],
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N)

    # num_of_episodes = 1
    # num_of_iterations = 1000

    score_history = np.zeros((num_of_episodes,))
    rewards = np.zeros((num_of_episodes, num_of_iterations))
    sumrate = np.zeros((num_of_episodes, num_of_iterations))
    U1_SINR = np.zeros((num_of_episodes, num_of_iterations))
    # U2_SINR = np.zeros((num_of_episodes, num_of_iterations))
    # users_sinr = np.zeros(
    #     (env.num_of_users, num_of_episodes, num_of_iterations))
    Old_Avg = 0
    obs = env.State()

    for ep in range(num_of_episodes):
        score = 0
        for iter in range(num_of_iterations):
            action = agent.choose_action(obs)
            new_state, reward, sumrate[ep][iter], SINRs = env.Step(action)
            agent.remember(obs, action, reward, new_state)
            agent.learn()
            obs = new_state
            score += reward
            rewards[ep][iter] = reward

            U1_SINR[ep][iter] = SINRs[0]
            # U2_SINR[ep][iter] = SINRs[1]

        score_history[ep] = score
        New_Avg = score_history[:ep + 1].mean()

        disp(episod=ep, score=score, score_history=score_history,
             New_Avg=New_Avg, Old_Avg=Old_Avg, last_u1_sinr=SINRs[0])

        obs = env.Reset()
        Old_Avg = New_Avg

    plot(score_history=score_history, sumrate=sumrate,
         u1_sinr=U1_SINR, u2_sinr=None, mean=False,
         title=f"N = {env.N}, M1 = {env.M1}, M2 = {env.M2}")

    agent.save_models()

    return score_history, sumrate, U1_SINR


if __name__ == "__main__":
    run()
