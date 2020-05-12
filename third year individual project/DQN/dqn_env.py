
import os
# from collections import namedtuple
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mp

# ShipState = namedtuple("ShipState", ["x", "y", "angle"])


def get_distance(state_a, state_b):
    dist = np.sqrt((state_a[0]-state_b[0])**2 + (state_a[1]-state_b[1])**2)
    return dist


def is_near(xy, left_down_pos, size, margin):
    if xy[0] >= (left_down_pos[0] - margin) \
       and xy[0] <= (left_down_pos[0] + size[0] + margin) \
       and xy[1] >= (left_down_pos[1] - margin) \
       and xy[1] <= (left_down_pos[1] + size[1] + margin):
        return True
    else:
        return False


def generate_env_randomly(map_size=100, num_obstacles=4, obstacles_size_range=(10, 20)):
    obstacle_pos_margin = 10
    obstacle_sizes = np.random.randint(*obstacles_size_range, size=(num_obstacles, 2))
    obstacle_positions = np.random.randint(obstacle_pos_margin, map_size-obstacle_pos_margin, size=(num_obstacles, 2))
    obstacles = {tuple(pos.tolist()):size.tolist() for pos, size in zip(obstacle_positions, obstacle_sizes)}

    # ship position limit
    pos_margin = 10
    pos_limit = 35
    # ship initial state
    is_initial_xy_legal = False 
    while not is_initial_xy_legal:
        initial_x = np.random.randint(pos_margin, map_size - pos_margin)
        if initial_x > pos_limit:
            initial_y = np.random.randint(pos_margin, pos_limit)
        else:
            initial_y = np.random.randint(pos_margin, map_size // 2)
        # initial place should be inside the obstacles
        is_initial_xy_legal = True
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            if is_near((initial_x, initial_y), pos, size, pos_margin):
                is_initial_xy_legal = False
                break

    # ship target state
    is_target_xy_legal = False
    while not is_target_xy_legal:
        target_x = np.random.randint(pos_margin, map_size - pos_margin)
        if target_x < (map_size - pos_limit):
            target_y = np.random.randint(map_size - pos_limit, map_size - pos_margin)
        else:
            target_y = np.random.randint(map_size // 2, map_size - pos_margin)
        # target place should be inside the obstacles
        is_target_xy_legal = True
        for pos, size in zip(obstacle_positions, obstacle_sizes):
            if is_near((target_x, target_y), pos, size, pos_margin):
                is_target_xy_legal = False

    return obstacles, (initial_x, initial_y, 90), (target_x, target_y, 0)


class Env():
    """the environment to define ship motion"""

    X_RANGE = 100
    Y_RANGE = 100
    BOAT_W = 3
    BOAT_L = 8

    def __init__(self,
            initial_ship_state,
            target_ship_state,
            obstacles,
            terminal_size=3,
            env_id=0):
        self.env_id = env_id
        self.max_dist = Env.X_RANGE + Env.Y_RANGE
        self.obstacles = obstacles
        self.initial_state = initial_ship_state
        self.target_state = target_ship_state
        self.terminal_size = terminal_size
        self.target_direction = self.calculate_target_direction_vector()
        print("env-{}, obstacles: {}, initial: {}, target: {}, direction: {}".format(
            self.env_id, self.obstacles, self.initial_state, self.target_state, self.target_direction))

        self.action_space = [-35, -15, 0, 15, 35]
        self.step = 15
        self.V = 0.19    # velocity for ship is assumed to be constant
        self.K = 0.08    # defines the turning capability coefficient
        self.T = 10.8    # defines the turning lag coefficient
        self.End = False

    def calculate_target_direction_vector(self):
        delta_pos = np.array([self.target_state[0]-self.initial_state[0], self.target_state[1]-self.initial_state[1]])
        # normalise to unit vectorr
        vector = delta_pos / np.sqrt(np.sum(np.power(delta_pos, 2)))
        return vector

    def choose_action_randomly(self):
        return np.random.choice(self.action_space)
        
    def obs_detect(self, state):
        """determines if the input state hits any obstacle, returns a boolean"""
        boolean = False
        ship_cx, ship_cy = state[:2]
        shape_half_size = max(Env.BOAT_W, Env.BOAT_L) // 2
        for pos, size in self.obstacles.items():
            if ship_cx > pos[0] - shape_half_size and ship_cx < pos[0] + size[0] + shape_half_size:
                if ship_cy > pos[1] - shape_half_size and ship_cy < pos[1] + size[1] + shape_half_size:
                    boolean = True
        return boolean
    
    def bound_detect(self, state):
        """determine if the input is out of bound"""
        boolean = False
        shape_half_size = max(Env.BOAT_W, Env.BOAT_L) // 2
        if (state[0]-shape_half_size)<0 or (state[0]+shape_half_size)>Env.X_RANGE:
            boolean = True
        if (state[1]-shape_half_size)<0 or (state[1]+shape_half_size)>Env.Y_RANGE:
            boolean = True
        return boolean

    def terminal_detect(self, state):
        delta_x = np.abs(state[0] - self.target_state[0])
        delta_y = np.abs(state[1] - self.target_state[1])
        if delta_x <= self.terminal_size and delta_y <= self.terminal_size:
            return True
        else:
            return False
    
    def distance_reward(self, state):
        dist = get_distance(state, self.target_state)
        # roughly between [-3.3, 0]
        return -0.03 * dist

    def direction_reward(self, state):
        ship_angle = state[2]
        angle_vec = np.array([np.cos(np.deg2rad(ship_angle)), np.sin(np.deg2rad(ship_angle))])
        dotval = np.sum(angle_vec * self.target_direction)
        # roughly between [-1, 0]
        return (dotval - 1)

    def nomoto_model(self, rudder_ang):
        """input a rudder angle, through the nomoto input-output model,
           returns the output ship course"
        """
        course = int(round(self.K * int(rudder_ang) * (self.step - self.T + self.T * np.exp(-self.step / self.T)), -1))
        return course

    def take_step(self, act_idx):
        ship_action = self.action_space[act_idx]
        ship_angle = (self.state[2] + self.nomoto_model(ship_action)) % 360
        ship_x = self.state[0] + self.V * np.cos(np.deg2rad(ship_angle)) * self.step
        ship_y = self.state[1] + self.V * np.sin(np.deg2rad(ship_angle)) * self.step
        self.state = [round(ship_x), round(ship_y), ship_angle]
        return self.check_state_reward()

    def check_state_reward(self):
        self.End = False
        if self.obs_detect(state=self.state):    #hits the obstacles
            reward = -1.0
            self.End = True
            # self.End = False
        elif self.bound_detect(state=self.state):  #out of bound
            reward = -1.0
            self.End = True
        else:
            dist_to_target = get_distance(self.state, self.target_state)
            if dist_to_target < self.terminal_size:
                reward = 1.0 - dist_to_target * 0.1
                self.End = True
            else:
                # distance reward function
                # r1 = self.distance_reward(self.state)
                # r1 = dist_to_target / Env.X_RANGE * 0.05
                # r2 = self.direction_reward(self.state) * 0.01 # roughly between [-1, 0] * 0.01
                # reward = r1 + r2
                reward = -0.015
                # print("rd: {:6.2f}, ra: {:6.2f}".format(r1, r2))
                self.End = False
        return reward, self.state, self.End


    def env_reset(self):
        # self.time = 0
        self.state = self.initial_state
        self.End = False
        return self.state
    
    def show(self, state_trace=[], action_list=[], is_random_act_list=[], reward_list=[], save_path="", prefix="", count=0):
        fig, axs = plt.subplots(figsize=(6, 6))
        axs.set_xlim(0, Env.X_RANGE)
        axs.set_ylim(0, Env.Y_RANGE)
        axs.set_facecolor('xkcd:sky blue')
        axs.scatter(self.initial_state[0], self.initial_state[1], marker='o', c='w', s=40)
        axs.scatter(self.target_state[0], self.target_state[1], marker='o', c='g', s=40)
        for pos, size in self.obstacles.items():
            rect = plt.Rectangle(pos, size[0], size[1], fc='r')
            axs.add_patch(rect)
        if len(state_trace) > 0:
            if len(is_random_act_list) > 0:
                exact_points = [s for s, r in zip(state_trace, is_random_act_list) if not r]
                axs.scatter([s[0] for s in exact_points], [s[1] for s in exact_points], marker='*', c='w', s=5)
                random_points = [s for s, r in zip(state_trace, is_random_act_list) if r]
                axs.scatter([s[0] for s in random_points], [s[1] for s in random_points], marker='x', c='r', s=5)
            else:
                axs.scatter([s[0] for s in state_trace], [s[1] for s in state_trace], marker='*', c='w', s=5)
            # draw ship shape, every 5 position
            for x, y, ang in state_trace[::5]:
                ship_m = mp.Ellipse((x,y), Env.BOAT_W, Env.BOAT_L, angle=ang-90, fill=False, ec='xkcd:purple')
                axs.add_patch(ship_m)
        # start position
        ship_s = mp.Ellipse(self.initial_state[:-1], Env.BOAT_W, Env.BOAT_L, angle=self.initial_state[2]-90, fill=False, ec='xkcd:purple')
        axs.add_patch(ship_s)
        # add string info
        title = ""
        if len(action_list) > 0:
            acts_str = "_".join([str(act) for act in action_list])
            title += acts_str + "\n"
        if len(reward_list) > 0:
            rewards_str = "tot: {:5.1f}_".format(np.sum(reward_list)) + "_".join(["{:5.2f}".format(r) for r in reward_list])
            title += rewards_str + "\n"
        if len(title) > 0:
            axs.set_title(title)
        if len(save_path) > 0:
            str_prefix = prefix + "_" if len(prefix) > 0 else ""
            save_name = os.path.join(save_path, "{}ep{:06d}_env{}.jpg".format(str_prefix, count, self.env_id))
            fig.savefig(save_name)
            print("Save img: {}".format(save_name))
        else:
            plt.show()


def test_main():
    obstacles, initial_ship_state, target_ship_state = generate_env_randomly()
    env = Env(initial_ship_state, target_ship_state, obstacles)
    env.show()


if __name__ == "__main__":
    
    test_main()
    print("Done")


"""
python environment.py
"""
