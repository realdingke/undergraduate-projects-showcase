
import os
import pickle
import datetime
import json
from distutils import dir_util

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from environment import Env, generate_env_randomly
from model import DQNModel, get_input_state, get_estimated_qvals, get_n_step_estimated_qvals, \
     choose_action, train_step, train_step_v0, eval_step, eval_step_v0, extend_input_state
from buffer import ReplayBuffer


def draw_vals(mean_ep_rewards, mean_loss_vals, per_num_envs, exp_prefix=""):
    assert len(mean_ep_rewards) == len(mean_loss_vals)
    xs = np.arange(len(mean_ep_rewards)) * per_num_envs
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # reward
    axs[0].plot(xs, mean_ep_rewards)
    axs[0].set_xlabel("episode")
    axs[0].set_title("rewards")
    # loss
    axs[1].plot(xs, mean_loss_vals)
    axs[1].set_xlabel("episode")
    axs[1].set_title("loss")
    str_prefix = exp_prefix + "_" if len(exp_prefix) > 0 else ""
    fig.savefig("{}training.jpg".format(str_prefix))


def run_one_episodes(
        env, model, epsilon, max_steps, without_map_info, random_initial_state_prob=0.0):
    ship_state_trace = []
    input_states = []
    action_list = []
    qval_list = []
    is_random_act_list = []
    reward_list = []
    done_list = []

    ship_state = env.env_reset()
    if np.random.rand() < random_initial_state_prob:
        initial_x = np.random.randint(0, env.map_size)
        initial_y = np.random.randint(0, env.map_size)
        initial_angle = np.random.choice(np.arange(0, 180, 10))
        env.state = (initial_x, initial_y, initial_angle)

    for _ in range(max_steps):
        ship_state_trace.append(ship_state)
        state_input = get_input_state(env, ship_state, without_map_info=without_map_info)
        input_states.append(state_input)
        # predict action
        act_taken, is_random_act, qval = choose_action(model, state_input[np.newaxis, :], epsilon)
        qval_list.append(qval)
        # take action
        reward, next_ship_state, done = env.take_step(act_taken)
        action_list.append(act_taken)
        is_random_act_list.append(int(is_random_act))
        reward_list.append(reward)
        done_list.append(int(done))
        ship_state = next_ship_state
        if done:
            break
    # the last state
    ship_state_trace.append(ship_state)
    state_input = get_input_state(env, ship_state, without_map_info=without_map_info)
    input_states.append(state_input)
    # the last qval
    if not done:
        qval = np.max(model(state_input[np.newaxis, :], training=False).numpy()[0])
    else:
        qval = 0
    qval_list.append(qval)
    return ship_state_trace, input_states, action_list, reward_list, done_list, is_random_act_list, np.array(qval_list)


def create_or_load_envs(
        num_envs=10,
        num_obstacls_ratio=[0.2, 0.3, 0.3, 0.2]):
    env_list = []
    if num_envs > 0:
        map_list = []
        num_env_respond_obstacls = [round(num_envs * r) for r in num_obstacls_ratio]
        print("num_env_respond_obstacls:", num_env_respond_obstacls)
        for nb, ne in enumerate(num_env_respond_obstacls):
            for k in range(ne):
                # randomly create a map
                obstacles, initial_ship_state, target_ship_state = generate_env_randomly(num_obstacles=nb)
                map_list.append((obstacles, initial_ship_state, target_ship_state))
                # create an evironment
                env = Env(initial_ship_state, target_ship_state, obstacles, env_id=len(env_list))
                env_list.append(env)
        pickle.dump(map_list, open("maps.pkl", "wb"))
        print("create {} envs done".format(num_envs))
    else:
        # reload ramdom maps
        map_list = pickle.load(open("maps.pkl", "rb"))
        for k in range(len(map_list)):
            # randomly create a map
            obstacles, initial_ship_state, target_ship_state = map_list[k]
            # create an evironment
            env = Env(initial_ship_state, target_ship_state, obstacles, env_id=len(env_list))
            env_list.append(env)
        print("load {} envs done".format(num_envs))
    return env_list


def train_main(
        exp_prefix="",
        fc_units=[128, 64, 64],
        env_list=[],
        num_envs=10,
        num_obstacls_ratio=[0.2, 0.3, 0.3, 0.2],
        n_step=1,
        max_episodes=10000,
        max_steps=120,
        per_num_envs=8,
        replay_buffer_len=400,
        no_replay=False,
        batch_size=64,
        learning_rate=1e-4,
        epsilon_min=0.05,
        epsilon_max=0.10,
        gamma=0.98,
        without_map_info=False,
        save_interval=1000,
        show=False):
    # create envs
    if len(env_list) == 0:
        env_list = create_or_load_envs(num_envs, num_obstacls_ratio)
    # create model
    if without_map_info:
        state_dims = 2 + 1
    else:
        state_dims = 4 * (2 + 2) + 6 + 2 + 2
    act_dims = 5
    model = DQNModel(state_dims=state_dims, act_dims=act_dims, fc_units=fc_units)
    print("create model done")
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    # create replay buffer
    buffer = ReplayBuffer(replay_buffer_len)
    print("create buffer done")

    # construct save path suffix
    weight_dir = os.path.join("weights", exp_prefix)
    dir_util.mkpath(weight_dir)
    log_dir = os.path.join("logs", exp_prefix)
    dir_util.mkpath(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # run simulations
    mean_loss_vals = []
    mean_ep_rewards = []
    last_save_ep_idx = 0
    for ep in range(max_episodes//per_num_envs):
        if no_replay:
            buffer.clear()
        num_new_samples = 0
        ep_rewards = []
        # randomly select an env and run rollout
        envs = np.random.choice(env_list, size=(per_num_envs))
        env_indices = np.random.randint(len(env_list), size=(per_num_envs))
        for roll_idx, env_idx in enumerate(env_indices):
            env = env_list[env_idx]
            episode_index = ep * per_num_envs + roll_idx
            epsilon = epsilon_max - (epsilon_max-epsilon_min)/max_episodes * episode_index
            ship_state_trace, input_states, action_list, reward_list, done_list, is_random_act_list, qvals = run_one_episodes(env, model, epsilon, max_steps, without_map_info)
            # td_errors = (reward_list + qvals[1:] * gamma) - qvals[:-1]
            td_errors = get_n_step_estimated_qvals(reward_list, qvals[1:], gamma, n_step) - qvals[:-1]
            buffer.add_items(input_states, action_list, reward_list, done_list, td_errors)
            num_new_samples += len(input_states)
            ep_rewards.append(np.sum(reward_list))
            print("episode {:4d}, env-{:03d}, epsilon: {:4.2f}, episode length: {:3d}, ep_reward: {:8.2f}".format(
                episode_index, env_idx, epsilon, len(input_states), np.sum(reward_list)))
            tot_ep_reward = np.sum(reward_list)
            avg_ep_reward = np.mean(reward_list)
            with summary_writer.as_default():
                tf.summary.scalar('tot_ep_reward_trn', tot_ep_reward, step=episode_index)
                tf.summary.scalar('avg_ep_reward_trn', avg_ep_reward, step=episode_index)
            if episode_index % 100 == 0:
                # run an evaluation
                (eval_ship_state_trace,
                 eval_input_states,
                 eval_action_list,
                 eval_reward_list,
                 eval_done_list,
                 eval_is_random_act_list,
                 eval_qval_list) = run_one_episodes(env, model, 0, max_steps, without_map_info)
                # log episode reward
                with summary_writer.as_default():
                    eval_tot_ep_reward = np.sum(eval_reward_list)
                    eval_avg_ep_reward = np.mean(eval_reward_list)
                    tf.summary.scalar('tot_ep_reward_evl', eval_tot_ep_reward, step=episode_index)
                    tf.summary.scalar('avg_ep_reward_evl', eval_avg_ep_reward, step=episode_index)
                # eval the loss
                eval_states_curr = np.array(eval_input_states[:-1])
                eval_states_next = np.array(eval_input_states[1:])
                eval_qvals_next = model(eval_states_next, training=False).numpy()
                eval_qvals_next_max = np.amax(eval_qvals_next, axis=1) * (1 - np.array(eval_done_list))
                eval_qvals_esti = get_n_step_estimated_qvals(eval_reward_list, eval_qvals_next_max, gamma, n_step)
                # to tensor
                eval_states_curr = tf.convert_to_tensor(eval_states_curr, tf.float32)
                eval_action_list_tf = tf.convert_to_tensor(eval_action_list)
                eval_qvals_esti = tf.convert_to_tensor(eval_qvals_esti, tf.float32)
                # eval to get loss
                eval_loss = eval_step_v0(model, eval_states_curr, eval_action_list_tf, eval_qvals_esti).numpy()
                with summary_writer.as_default():
                    tf.summary.scalar('loss_evl', eval_loss, step=episode_index)
                # draw map and state trace
                env.show(
                    eval_ship_state_trace,
                    np.sum(eval_reward_list),
                    eval_loss,
                    eval_action_list,
                    eval_is_random_act_list,
                    save_path="pictures",
                    prefix=exp_prefix,
                    count=episode_index)
        # run update
        avg_ep_reward = float(np.mean(ep_rewards))
        mean_ep_rewards.append(avg_ep_reward)
        curr_update_loss_vals = []
        if no_replay:
            num_updates = 1
        else:
            num_updates = max(1, min(num_new_samples, replay_buffer_len)//batch_size)
        for _ in range(num_updates):
            # get qvals of next states
            if no_replay:
                batch_size = max(1, int(num_new_samples * 0.8))  # overwrite batch_size
            states_curr, states_next, actions, rewards, dones = buffer.sample(batch_size)
            states_next = tf.convert_to_tensor(states_next, tf.float32)
            qvals_next = model(states_next, training=False).numpy()
            qvals_next = np.amax(qvals_next, axis=1) * (1 - dones)
            qvals_esti = get_n_step_estimated_qvals(rewards, qvals_next, gamma, n_step)
            # to tensor 
            states_curr = tf.convert_to_tensor(states_curr, tf.float32)
            actions = tf.convert_to_tensor(actions)
            qvals_esti = tf.convert_to_tensor(qvals_esti, tf.float32)
            # do an update
            loss_trn = train_step_v0(model, optimizer, states_curr, actions, qvals_esti).numpy()
            with summary_writer.as_default():
                tf.summary.scalar('loss_trn', loss_trn, step=episode_index)
            curr_update_loss_vals.append(loss_trn)
            print("episode {:4d}, bs: {:4d}, loss_trn: {:6.2f}".format(episode_index, batch_size, loss_trn))
        mean_loss_vals.append(float(np.mean(curr_update_loss_vals)))

        # draw loss
        if ep > 0 and ep % 10 == 0:
            draw_vals(mean_ep_rewards, mean_loss_vals, per_num_envs, exp_prefix=exp_prefix)
            # save to file for further use
            json.dump([mean_loss_vals, mean_ep_rewards], open("logs/{}_logs_info.json".format(exp_prefix), "w"))

        # Save the weights using the `checkpoint_path` format
        if (episode_index - last_save_ep_idx) > save_interval:
            save_path = os.path.join(weight_dir, "weights_{:05d}.ckpt".format(episode_index))
            model.save_weights(save_path)
            last_save_ep_idx = episode_index
            print("episode-{}, save weights to: {}".format(episode_index, save_path))


def multi_map_experiment():
    # multi obs
    train_main(
        exp_prefix="multi_sim",
        fc_units=[128, 64, 64],
        num_envs=40,
        num_obstacls_ratio=[0.1, 0.1, 0.2, 0.2, 0.4],
        max_episodes=20000,
        per_num_envs=8,
        replay_buffer_len=800,
        batch_size=256,
        epsilon_min=0.04,
        epsilon_max=0.12,
        without_map_info=False,
        show=False
    )


def single_map_experiment(net_size="small"):
    # specify the env
    # obstacles = {(20, 10): [50, 15]}
    # initial_ship_state = (80, 10, 90)
    # target_ship_state = (90, 90, 0)
    # env = Env(initial_ship_state, target_ship_state, obstacles)

    # specify the env
    obstacles = {(30, 45): [45, 15]}
    initial_ship_state = (15, 10, 90)
    target_ship_state = (90, 90, 0)
    env = Env(initial_ship_state, target_ship_state, obstacles)

    # 200 size map
    # obstacles = {(60, 90): [45, 15]}  # double position, but same size
    # initial_ship_state = (15, 10, 90)
    # target_ship_state = (190, 190, 0)
    # env = Env(initial_ship_state, target_ship_state, obstacles, map_size=200)

    if 0:
        train_main(
            exp_prefix="sin_map_run10",
            fc_units=[128, 64, 64],
            without_map_info=True,
            env_list=[env],
            max_episodes=40000,
            per_num_envs=8,
            replay_buffer_len=400,
            learning_rate=0.0002,
            batch_size=64,
            epsilon_min=0.05,
            epsilon_max=0.20,
            show=False
        )
    if 0:
        train_main(
            exp_prefix="sin_map_{}_run4-1".format(net_size),
            fc_units=[64, 32, 16],
            without_map_info=True,
            env_list=[env],
            max_episodes=20000,
            per_num_envs=1,
            replay_buffer_len=200,
            learning_rate=0.0004,
            batch_size=64,
            epsilon_min=0.05,
            epsilon_max=0.15,
            show=False
        )
    if 0:
        train_main(
            exp_prefix="blocked_{}_run3_fix_r1_no_r2_ns8".format(net_size),
            fc_units=[64, 64, 64],
            n_step=8,
            without_map_info=True,
            env_list=[env],
            max_episodes=50000,
            per_num_envs=8,
            replay_buffer_len=500,
            learning_rate=0.0004,
            batch_size=128,
            epsilon_min=0.05,
            epsilon_max=0.15,
            show=False
        )
    if 0:
        # vs run3_fix_r1_no_r2_ns8, smaller epsilon, per_num_envs
        train_main(
            exp_prefix="blocked_{}_run4_fix_r1_no_r2_ns8".format(net_size),
            fc_units=[64, 64, 64],
            n_step=8,
            without_map_info=True,
            env_list=[env],
            max_episodes=40000,
            per_num_envs=4,
            replay_buffer_len=200,
            learning_rate=0.0004,
            batch_size=64,
            epsilon_min=0.03,
            epsilon_max=0.10,
            show=False
        )
    if 0:
        # vs run4_fix_r1_no_r2_ns8, has r2
        train_main(
            exp_prefix="blocked_{}_run5_fix_r1_small_r2_ns8".format(net_size),
            fc_units=[64, 64, 64],
            n_step=8,
            without_map_info=True,
            env_list=[env],
            max_episodes=40000,
            per_num_envs=4,
            replay_buffer_len=200,
            learning_rate=0.0004,
            batch_size=64,
            epsilon_min=0.03,
            epsilon_max=0.10,
            show=False
        )
    if 0:
        # vs large network, longer n step <-- bad, hurge loss
        train_main(
            exp_prefix="blocked_{}_run6_fix_r1_small_r2_ns8".format(net_size),
            fc_units=[128, 128, 64],
            n_step=16,
            without_map_info=True,
            env_list=[env],
            max_episodes=50000,
            per_num_envs=4,
            replay_buffer_len=200,
            learning_rate=0.0004,
            batch_size=64,
            epsilon_min=0.03,
            epsilon_max=0.10,
            show=False
        )
    if 0:
        # vs run3, ship state as 4 value, smaller net
        train_main(
            exp_prefix="blocked_{}_run7_fix_r1_no_r2_ns8".format(net_size),
            fc_units=[32, 16, 16],
            n_step=8,
            without_map_info=True,
            env_list=[env],
            max_episodes=50000,
            per_num_envs=8,
            replay_buffer_len=500,
            learning_rate=0.0004,
            batch_size=128,
            epsilon_min=0.05,
            epsilon_max=0.15,
            show=False
        )
    if 1:
        # run8 vs run3, ship state as 4 value, smaller net
        # run9 vs run8, add bonus for stopping near to target
        # run10 vs run9, no replay
        # run11 vs run9, bigger network
        # run12 vs run9, 200 size map
        # run10_rerun vs run9, rerun, nothing change, !!! name should be run9_rerun
        # run13 vs run9, buffer_len: 250 -> 500. (run9 buffer len should be 500?)
        # run14 vs run13, lr 0.0004 -> 0.001
        # run14 vs run14, ship state as 3 value, !!! should be run15
        # run16 vs run15, lr 0.0004 -> 0.001, inp3, size median, 
        train_main(
            exp_prefix="blocked_median_run16_fix_r12_no_r2_ns8_dis_3inp",
            fc_units=[64, 32, 32],
            n_step=8,
            no_replay=False,
            without_map_info=True,
            env_list=[env],
            max_episodes=50000,
            per_num_envs=8,
            replay_buffer_len=500,
            learning_rate=0.0001,
            batch_size=128,
            epsilon_min=0.05,
            epsilon_max=0.20,
            show=False
        )


if __name__ == "__main__":

    single_map_experiment(net_size="small")
    # single_map_experiment(net_size="median")


"""
python train.py
"""
