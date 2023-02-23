import streamlit as st
from Environment import Environment
from main import run

st.header("Environment Variables")
col1, col2 = st.columns(2)

num_of_antennas = col1.text_input("Number of Antennas", 5)
num_of_irs1 = col2.text_input("Number of IRS1 Elements", 5)
num_of_irs2 = col1.text_input("Number of IRS2 Elements", 5)
path_loss_exponent = col2.text_input("Path Loss Exponent", 2)
irs1_to_antenna = col1.text_input("IRS1 to Antenna Distance", 20)
irs2_to_antenna = col2.text_input("IRS2 to Antenna Distance", 20)
irs1_to_irs2 = col1.text_input("IRS1 to IRS2 Distance", 10)
# num_of_users = col2.text_input("Number of Users", 1)

env = Environment(num_of_antennas=int(num_of_antennas), num_of_irs1=int(num_of_irs1),
                  num_of_irs2=int(num_of_irs2),
                  path_loss_exponent=float(path_loss_exponent), irs1_to_antenna=float(irs1_to_antenna),
                  irs2_to_antenna=float(irs2_to_antenna), irs1_to_irs2=float(irs1_to_irs2))


with st.expander("User 1 Variables"):
    with st.form("User 1 Variables"):
        col1, col2 = st.columns(2)
        distance_to_antenna = col1.text_input("Distance to Antenna", 40)
        distance_to_irs1 = col2.text_input("Distance to IRS1", 10)
        distance_to_irs2 = col1.text_input("Distance to IRS2", 20)
        noise_var = col2.text_input("Noise Variance", 1e-4)
        los_to_antenna = col1.text_input("LOS to Antenna", False)
        los_to_irs1 = col2.text_input("LOS to IRS1", True)
        los_to_irs2 = col1.text_input("LOS to IRS2", False)
        sinr_threshold = col2.text_input("SINR Threshold", 1)
        penalty = col1.text_input("Penalty", 0)
        allocated_power = col2.text_input("Allocated Power", 1)
        weight = st.text_input("Weight", 1)
        submitted = st.form_submit_button("Submit")


U1 = env.CreateUser(distance_to_antenna=float(distance_to_antenna), distance_to_irs1=float(distance_to_irs1),
                    distance_to_irs2=float(distance_to_irs2), noise_var=float(noise_var),
                    los_to_antenna=bool(los_to_antenna), los_to_irs1=bool(los_to_irs1),
                    los_to_irs2=bool(los_to_irs2), sinr_threshold=float(sinr_threshold),
                    penalty=float(penalty), allocated_power=float(allocated_power), weight=float(weight))

with st.expander("User 2 Variables"):
    with st.form("User 2 Variables"):
        col1, col2 = st.columns(2)
        distance_to_antenna = col1.text_input("Distance to Antenna", 40)
        distance_to_irs1 = col2.text_input("Distance to IRS1", 20)
        distance_to_irs2 = col1.text_input("Distance to IRS2", 10)
        noise_var = col2.text_input("Noise Variance", 1e-4)
        los_to_antenna = col1.text_input("LOS to Antenna", True)
        los_to_irs1 = col2.text_input("LOS to IRS1", True)
        los_to_irs2 = col1.text_input("LOS to IRS2", True)
        sinr_threshold = col2.text_input("SINR Threshold", 1)
        penalty = col1.text_input("Penalty", 0)
        allocated_power = col2.text_input("Allocated Power", 1)
        weight = st.text_input("Weight", 1)
        submitted = st.form_submit_button("Submit")

U2 = env.CreateUser(distance_to_antenna=float(distance_to_antenna), distance_to_irs1=float(distance_to_irs1),
                    distance_to_irs2=float(distance_to_irs2), noise_var=float(noise_var),
                    los_to_antenna=bool(los_to_antenna), los_to_irs1=bool(los_to_irs1),
                    los_to_irs2=bool(los_to_irs2), sinr_threshold=float(sinr_threshold),
                    penalty=float(penalty), allocated_power=float(allocated_power), weight=float(weight))

st.header("Run Time Variables")
with st.form("Run Time Variables"):
    col1, col2 = st.columns(2)
    num_of_iterations = col2.text_input("Number of Iterations", 150)
    num_of_episodes = col1.text_input("Number of Episodes", 150)
    # mean_reward = col1.text_input(
    #     "Mean Reward", placeholder="0 for False, 1 for True")
    submitted = st.form_submit_button("Submit")

run_button = st.button("Run")

if run_button:
    run(env=env, num_of_iterations=int(num_of_iterations),
        num_of_episodes=int(num_of_episodes))
