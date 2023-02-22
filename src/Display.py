from termcolor import colored


def disp(*, episod, score, score_history, New_Avg, Old_Avg, SINRs, sumrate):
    if score >= Old_Avg:
        tmp_str = f"{New_Avg: < 10.2f}" + colored(" +", "green")

    else:
        tmp_str = f"{New_Avg: < 10.2f}" + colored(" -", "red")

    if score == max(score_history[:episod + 1]):
        tmp_str += " Max " + "\U0001f600"
        score_string = colored(f"{score: < 10.2f}", "green")
    else:
        tmp_str += "       "
        score_string = f"{score: < 10.2f}"

    print(f"Episode{episod + 1: < 4}", f"Score -> {score_string}",
          f"Avg-Score -> {tmp_str}", f"U1-SINR -> {SINRs[0]: < 8.2f}",
          f"U2-SINR -> {SINRs[1]: < 8.2f}",
          f"Sumrate -> {sumrate: < 8.2f}")
