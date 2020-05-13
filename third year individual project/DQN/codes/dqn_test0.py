
import os
import json

import matplotlib.pyplot as plt


def run_main():
    # load json file
    exp_names = [
        "sin_map_small_net_run1",
        "sin_map_small_net_run2",
        "sin_map_median_net_run1",
        "sin_map_small_net_run1_adam"
    ]

    loss_vals_list = []
    ep_rewards_list = []
    for exp in exp_names:
        filename = os.path.join("logs", "{}_logs_info.json".format(exp))
        mean_loss_vals, mean_ep_rewards = json.load(open(filename, "r"))
        loss_vals_list.append(mean_loss_vals)
        ep_rewards_list.append(mean_ep_rewards)


if __name__ == "__main__":

    run_main()
    print("Done")

