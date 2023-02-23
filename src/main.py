from Environment import Environment
from Agent import Agent
import numpy as np
from plot import plot
from Display import disp
import streamlit as st


def run(env, num_of_episodes, num_of_iterations, mean_reward):

    agent = Agent(num_states=env.num_of_users, bound=2, batch_size=64, max_size=100000,
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N,
                  noise=0.02, alpha=0.0002, beta=0.0004, fc1=1024, fc2=512)

    score_history = np.zeros((num_of_episodes,))
    rewards = np.zeros((num_of_episodes, num_of_iterations))
    sumrate = np.zeros((num_of_episodes, num_of_iterations))
    U1_SINR = np.zeros((num_of_episodes, num_of_iterations))

    U2_SINR = np.zeros((num_of_episodes, num_of_iterations))

    Old_Avg = 0
    obs = env.State()

    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)

    for ep in range(num_of_episodes):
        score = 0
        obs = env.State()
        for iter in range(num_of_iterations):
            action = agent.choose_action(obs)

            new_state, reward, sumrate[ep][iter], SINRs = env.Step(action)
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
             New_Avg=New_Avg, Old_Avg=Old_Avg)

        # obs = env.Reset()
        Old_Avg = New_Avg

        my_bar.progress((ep + 1) / num_of_episodes, text=progress_text)

    st.header("Results")
    plot(score_history=score_history, sumrate=sumrate,
         u1_sinr=U1_SINR, u2_sinr=U2_SINR, mean=False,
         title=f"N = {env.N}, M1 = {env.M1}, M2 = {env.M2}")

    # agent.save_models()
