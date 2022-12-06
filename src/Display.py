from termcolor import colored


def disp(*, episod, score, score_history, New_Avg, Old_Avg):
    if score >= Old_Avg:
        tmp_str = f"{New_Avg: < 10.2f}" + colored(" +", "green")

    else:
        tmp_str = f"{New_Avg: < 10.2f}" + colored(" -", "red")

    if score == max(score_history[:episod + 1]):
        tmp_str += " Max " + "\U0001f600"
        score_string = colored(f"{score: < 10.2f}", "green")
    else:
        score_string = f"{score: < 10.2f}"

    print(f"Episode{episod + 1: < 4}", f"Score -> {score_string}",
          f"Avg-Score -> {tmp_str}")
