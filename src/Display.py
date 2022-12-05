from termcolor import colored


def disp(*, episod, score, score_history, New_Avg, Old_Avg):
    if score >= Old_Avg:
        tmp_str = f"{New_Avg: < 10.2f}  +"

    else:
        tmp_str = f"{New_Avg: < 10.2f}  -"

    if score == max(score_history):
        tmp_str += " Max :)"
        score_string = colored(f"{score: < 10.2f}", "green")
    else:
        score_string = f"{score: < 10.2f}"

    print(f"Episode{episod + 1: < 4}", f"Score -> {score_string}",
          f"Avg-Score -> {tmp_str}")
