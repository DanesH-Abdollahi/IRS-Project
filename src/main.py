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
                        los_to_irs2=False, sinr_threshold=1, penalty=0, allocated_power=1, weight=1)

    U2 = env.CreateUser(distance_to_antenna=40, distance_to_irs1=20, distance_to_irs2=10,
                        noise_var=1e-4, los_to_antenna=True, los_to_irs1=True,
                        los_to_irs2=True, sinr_threshold=1, penalty=0, allocated_power=1, weight=1)

    agent = Agent(num_states=env.num_of_users, bound=2, batch_size=64, max_size=100000,
                  env=env, n_actions=env.M1 + env.M2 + len(env.Users) * env.N,
                  noise=0.02, alpha=0.0002, beta=0.0004, fc1=1024, fc2=512)

    env.Hs1 = np.array([[-0.04104735+0.02855022j,  0.01969476-0.04595777j,
                         -0.02084386+0.04544814j,  0.04927599+0.00847804j,
                         0.03666268+0.03399776j],
                        [-0.04577318-0.02012003j, -0.04926676-0.0085315j,
                         0.02241783-0.04469274j,  0.04438554+0.02302008j,
                         -0.00569801+0.04967427j],
                        [-0.01868735-0.04637653j, -0.00530821-0.04971743j,
                         -0.01776597-0.04673725j,  0.02801236+0.04141627j,
                         0.00485043+0.04976418j],
                        [-0.04028231+0.02961985j, -0.04158905+0.0277552j,
                         0.04403966+0.02367506j, -0.04992086+0.00281203j,
                         -0.02595928-0.04273307j],
                        [0.04464509-0.02251258j,  0.03830753+0.03213305j,
                         -0.04304204-0.02544371j, -0.04962215+0.00613536j,
                         0.02112823+0.04531664j]])

    env.Hs2 = np.array([[-0.03746477+0.03311179j,  0.03957882+0.03055351j,
                         -0.03083897+0.0393568j,  0.04879403-0.01091523j,
                         0.02218462-0.04480896j],
                        [0.02579013+0.04283538j,  0.02464726-0.04350302j,
                         0.04756426+0.01541562j,  0.02470188-0.04347203j,
                         0.02971186+0.04021449j],
                        [-0.04968244-0.00562626j, -0.02533401+0.04310671j,
                         0.01990123+0.04586874j,  0.04275847+0.02591743j,
                         -0.04102392+0.02858388j],
                        [0.01748259-0.04684399j, -0.04952994+0.00683997j,
                         -0.01070854+0.04883981j, -0.04910326-0.00942707j,
                         -0.02095297+0.04539794j],
                        [-0.01467585-0.04779769j, -0.03386839+0.03678223j,
                         0.01591529-0.0473994j, -0.01942553-0.04607221j,
                         -0.036294 + 0.03439106j]])

    env.H12 = np.array([[-0.00553294-0.09984682j, -0.02791645-0.09602433j,
                         0.07649098-0.06441374j, -0.09116592-0.04109471j,
                         0.0990288 + 0.01390313j],
                        [0.05284954-0.08489362j,  0.046967 + 0.08828421j,
                         -0.09381925+0.03461138j,  0.02022456+0.09793348j,
                         0.07490933-0.06624645j],
                        [0.05888394+0.08082501j,  0.09372589-0.03486341j,
                         0.09517965+0.03067303j, -0.07562102-0.06543288j,
                         0.07706626+0.06372434j],
                        [0.02718112-0.09623506j, -0.0135 + 0.09908456j,
                         -0.08746539-0.0484748j, -0.05417964+0.08405098j,
                         0.09999991-0.00013529j],
                        [-0.07397138-0.06729216j, -0.03705068-0.09288297j,
                         -0.04007744+0.09161768j, -0.0644333 + 0.0764745j,
                         0.09167526+0.03994554j]])

    env.H21 = np.array([[-0.00553294+0.09984682j,  0.05284954+0.08489362j,
                         0.05888394-0.08082501j,  0.02718112+0.09623506j,
                         -0.07397138+0.06729216j],
                        [-0.02791645+0.09602433j,  0.046967 - 0.08828421j,
                         0.09372589+0.03486341j, -0.0135 - 0.09908456j,
                         -0.03705068+0.09288297j],
                        [0.07649098+0.06441374j, -0.09381925-0.03461138j,
                         0.09517965-0.03067303j, -0.08746539+0.0484748j,
                         -0.04007744-0.09161768j],
                        [-0.09116592+0.04109471j,  0.02022456-0.09793348j,
                         -0.07562102+0.06543288j, -0.05417964-0.08405098j,
                         -0.0644333 - 0.0764745j],
                        [0.0990288 - 0.01390313j,  0.07490933+0.06624645j,
                         0.07706626-0.06372434j,  0.09999991+0.00013529j,
                         0.09167526-0.03994554j]])

    U1.hsu = np.array([[-0.02406356+0.00677829j,  0.02005993-0.01491977j,
                        -0.02464539+0.00419579j, -0.01101192+0.0224441j,
                        0.02088535-0.01374053j]])

    U1.h1u = np.array([[0.08419902+0.05394929j,  0.02275322+0.09737706j,
                        -0.06872893-0.07263838j,  0.08475439-0.05307253j,
                        -0.03336958+0.09426808j]])

    U1.h2u = np.array([[-0.04713141+0.01669221j,  0.04162465+0.02770178j,
                        0.04663206-0.01804025j, -0.00205603-0.04995771j,
                        0.04902818-0.00981009j]])

    U2.hsu = np.array([[0.02420573-0.00625163j, -0.00634876+0.02418043j,
                        -0.0129695 - 0.02137269j, -0.00541226-0.02440712j,
                        0.00794379+0.02370435j]])

    U2.h1u = np.array([[-0.02846405-0.04110715j,  0.0034285 - 0.04988232j,
                        -0.02376696+0.04399013j,  0.00195397-0.04996181j,
                        -0.03203853-0.03838662j]])

    U2.h2u = np.array([[-7.21171298e-03+0.09973962j,  9.54250025e-02-0.02990098j,
                        -4.34660973e-05+0.09999999j,  7.74890757e-02+0.06320952j,
                        -3.29208935e-02-0.09442571j]])

    num_of_episodes = 150
    num_of_iterations = 200

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

        if ep < num_of_episodes / 3:
            agent.noise = 0.25
        elif ep < num_of_episodes * 2 / 3:
            agent.noise = 0.10
        else:
            agent.noise = 0.055

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
