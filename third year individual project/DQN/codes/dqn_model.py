
import datetime
import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import Model


def get_input_state(env, ship_state, max_num_obstacles=4, without_map_info=False):
    # map info
    map_size = env.map_size
    ship_pos, ship_angle = np.array(ship_state[:2]), ship_state[2]
    if not without_map_info:
        # construct obstacles represent, shape: [N, max_num_obstacles * 4]
        obstacles_reps = []
        for pos, size in env.obstacles.items():
            # the relative position of obstacles' [x_l, x_r, y_b, y_u]
            rep = np.array([
                pos[0] - ship_pos[0],
                pos[0] + size[0] - ship_pos[0],
                pos[1] - ship_pos[1],
                pos[1] + size[1] - ship_pos[1]])
            obstacles_reps.append(rep)
        while len(obstacles_reps) < max_num_obstacles:
            # pad values
            rep = np.array([map_size, map_size, map_size, map_size])
            obstacles_reps.append(rep)
        # normalize position
        obstacles_reps = np.array(obstacles_reps).flatten() / map_size
    # angle to vector
    angle_vector = [np.cos(np.deg2rad(ship_angle)), np.sin(np.deg2rad(ship_angle))]
    # final representation
    if not without_map_info:
        positions_rep = np.concatenate([env.initial_state[:2], env.target_state[:2], ship_state[:2]]) / map_size
        target_delta_pos = (np.array(env.target_state[:2]) - ship_pos) / map_size
        represent = np.concatenate([obstacles_reps, positions_rep, target_delta_pos, angle_vector])
    else:
        # ship position and angle info
        # represent = np.concatenate([np.array(ship_state[:2])/map_size, angle_vector])
        represent = np.array([ship_state[0]/map_size, ship_state[1]/map_size, ship_state[2]/360])
    return represent.astype(np.float32)


def extend_input_state(input_states, max_num_obstacles=4):
    num_obstacles_reps = max_num_obstacles * 4
    obstacles_reps, other_reps = input_states[:, :num_obstacles_reps], input_states[:, 4*max_num_obstacles:]
    obstacles_reps = obstacles_reps.reshape(-1, max_num_obstacles, 4)
    random_indices = np.arange(max_num_obstacles)
    np.random.shuffle(random_indices)
    obstacles_reps = obstacles_reps[:, random_indices, :]
    obstacles_reps = obstacles_reps.reshape(-1, num_obstacles_reps)
    reps = np.concatenate([obstacles_reps, other_reps], axis=1)
    return reps


class DQNModel(Model):

    def __init__(self, state_dims, act_dims, fc_units=[512, 256, 128]):
        super(DQNModel, self).__init__()
        self.state_dims = state_dims
        self.act_dims = act_dims
        self.fc0 = Dense(fc_units[0], input_shape=(state_dims,), activation="relu", name="fc0")
        self.fc1 = Dense(fc_units[1], activation="relu", name="fc1")
        self.fc2 = Dense(fc_units[2], activation="relu", name="fc2")
        self.fc3 = Dense(act_dims, activation=None, name="fc3")
        self.bn0 = BatchNormalization(name="bn0")
        self.bn1 = BatchNormalization(name="bn1")
        # self.bn2 = BatchNormalization(name="bn2")

    def call(self, x):
        x = self.fc0(x)
        x = self.bn0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.fc3(x)
        return x


def choose_action(model, state_inputs, epsilon=0.1):
    """ inference the action index
    state_inputs.shape: [1, state_dims]
    """
    Q_vals = model(state_inputs, training=False).numpy()[0]
    if np.random.rand() <= epsilon:
        act_idx = np.random.randint(0, model.act_dims) # random action
        return act_idx, True, np.max(Q_vals)
    else:
        act_idx = np.argmax(Q_vals)
        return act_idx, False, np.max(Q_vals)


def get_estimated_qvals(model, rewards, states_next, gamma, n_step):
    qvals_next = model(states_next, training=False).numpy()
    qvals_esti = np.array(rewards) + np.amax(qvals_next, axis=1) * gamma
    return qvals_esti.astype(np.float32)


def get_n_step_estimated_qvals(rewards, qvals_next_max, gamma, n_step):
    assert len(rewards) == len(qvals_next_max), "error: {}, {}".format(rewards, qvals_next_max)
    steps = min(n_step, len(rewards))
    if not isinstance(rewards, np.ndarray):
        rewards = np.array(rewards)
    if not isinstance(qvals_next_max, np.ndarray):
        qvals_next_max = np.array(qvals_next_max)
    r_powers = np.zeros(shape=(steps, len(rewards)))
    r_powers[0] = rewards
    for k in range(1, steps):
        r_powers[k][:-k] = rewards[k:] * np.power(gamma, k)
    # get gamma^steps * qvals_next_max
    shifted_qvals_next_max = np.zeros_like(qvals_next_max, dtype=np.float32)
    if steps > 1:
        shifted_qvals_next_max[:-(steps-1)] = np.power(gamma, steps) * qvals_next_max[steps-1:]
        for i in range(1, steps):
            shifted_qvals_next_max[-i] = qvals_next_max[-1] * np.power(gamma, i)
    else:
        shifted_qvals_next_max = gamma * qvals_next_max

    qvals_esti = np.sum(r_powers, axis=0) + shifted_qvals_next_max
    return qvals_esti

    # discounted rewards one line
    # lfilter(b, a, x, axis=-1, zi=None)
    # a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
    #                       - a[1]*y[n-1] - ... - a[N]*y[n-N]
    # y[n] = x[n] + gamma * y[n-1]
    # y[n] = x[n] + x[n-1] + x[n-2] ... + gamma
    # discount = lambda x, gamma: lfilter([1], [1, -gamma], x[::-1])[::-1]


# @tf.function
def train_step_v0(model, optimizer, states_curr, actions, qvals_esti):
    with tf.GradientTape() as tape:
        qvals_pred = model(states_curr, training=True)
        act_qvals = tf.reduce_sum(qvals_pred * tf.one_hot(indices=actions, depth=model.act_dims), axis=1)
        # minimize the td error
        td_error = qvals_esti - act_qvals
        loss = tf.reduce_mean(tf.square(td_error))
        # clip loss
        # loss = tf.clip_by_value(loss, -50, 50)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 4.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# @tf.function
def train_step(model, optimizer, states_curr, actions, rewards, qvals_next, gamma):
    with tf.GradientTape() as tape:
        qvals_pred = model(states_curr, training=True)
        disbalance = tf.reduce_mean(tf.square(qvals_pred - tf.reduce_mean(qvals_pred, axis=1, keepdims=True)))
        act_qvals = tf.reduce_sum(qvals_pred * tf.one_hot(indices=actions, depth=model.act_dims), axis=1)
        # minimize the td error
        td_error = rewards + gamma * (qvals_next - act_qvals)
        td_loss = tf.reduce_mean(tf.square(td_error))
        loss = td_loss + disbalance * 0.01
        # clip loss
        # loss = tf.clip_by_value(loss, -50, 50)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 4.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def eval_step_v0(model, states_curr, actions, qvals_esti):
    qvals_pred = model(states_curr, training=True)
    act_qvals = tf.reduce_sum(qvals_pred * tf.one_hot(indices=actions, depth=model.act_dims), axis=1)
    # minimize the td error
    td_error = qvals_esti - act_qvals
    loss = tf.reduce_mean(tf.square(td_error))
    # clip loss
    loss = tf.clip_by_value(loss, -50, 50)
    return loss


def eval_step(model, states_curr, actions, rewards, qvals_next, gamma):
    qvals_pred = model(states_curr, training=True)
    act_qvals = tf.reduce_sum(qvals_pred * tf.one_hot(indices=actions, depth=model.act_dims), axis=1)
    # minimize the td error
    td_error = rewards + gamma * qvals_next - act_qvals
    loss = tf.reduce_mean(tf.square(td_error))
    # clip loss
    loss = tf.clip_by_value(loss, -50, 50)
    return loss


def test_main():

    if 0:
        input_states = np.arange(4*20).reshape(4, 20)
        print(input_states)
        reps = extend_input_state(input_states, max_num_obstacles=4)
        print(reps)

        input_states = np.arange(4*10).reshape(4, 10)
        print(input_states)
        reps = extend_input_state(input_states, max_num_obstacles=2)
        print(reps)

    state_dims = 4 * 4 + 3
    act_dims = 5
    model = DQNModel(state_dims, act_dims)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-4)
    # create test data
    batch_size = 2
    # test_x = np.random.uniform(size=(batch_size, state_dims))
    states_curr = tf.random.normal(shape=(batch_size, state_dims), name="states_curr")
    actions = tf.random.uniform(shape=[batch_size,], minval=0, maxval=act_dims, dtype=tf.int32, name="actions") 
    rewards = tf.random.normal(shape=(batch_size,), name="rewards")
    qvals_next = tf.random.normal(shape=(batch_size,), name="qvals_next")
    gamma = tf.constant(0.98, name="gamma")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = 'logs/func/%s' % stamp
    writer = tf.summary.create_file_writer(logdir)

    # Bracket the function call with
    # tf.summary.trace_on() and tf.summary.trace_export().
    tf.summary.trace_on(graph=True, profiler=True)
    # Call only one tf.function when tracing.
    train_step(model, optimizer, states_curr, actions, rewards, qvals_next, gamma)
    with writer.as_default():
        tf.summary.trace_export(
            name="train_step_trace",
            step=0,
            profiler_outdir=logdir)


def create_graph():
    state_dims = 6 + 2
    act_dims = 5
    batch_size = 2

    # model = DQNModel(state_dims, act_dims)
    # Define the model.
    fc_units = [64, 32, 32]
    model = tf.keras.models.Sequential([
        Dense(fc_units[0], input_shape=(state_dims,), activation="relu", name="fc0"),
        BatchNormalization(name="bn0"),
        Dense(fc_units[1], activation="relu", name="fc1"),
        BatchNormalization(name="bn1"),
        Dense(fc_units[1], activation="relu", name="fc2"),
        Dense(act_dims, activation=None, name="Qvals"),
    ])
    model.compile(
        optimizer='sgd',
        loss='mean_squared_error')
    # mean_square_error

    logdir = "test/"
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    states_curr = tf.random.normal(shape=(batch_size, state_dims), name="states_curr")
    qvals_esti = tf.random.normal(shape=(batch_size,), name="qvals_esti")

    # Train the model.
    model.fit(
        states_curr,
        qvals_esti, 
        batch_size=batch_size,
        epochs=1, 
        callbacks=[tensorboard_callback])

if __name__ == "__main__":

    # test_main()
    # create_graph()

    rewards = [1, 2, 3, 4, 5, 6]
    qvals_next_max = [0, 1, 2, 3, 4, 5]
    gamma = 0.5
    n_step = 4
    qvals_esti = get_n_step_estimated_qvals(rewards, qvals_next_max, gamma, n_step)
    print(qvals_esti)

    n_step = 1
    qvals_esti = get_n_step_estimated_qvals(rewards, qvals_next_max, gamma, n_step)
    print(qvals_esti)

    rewards = [1, 2]
    qvals_next_max = [1, 2]
    gamma = 0.5
    n_step = 4
    qvals_esti = get_n_step_estimated_qvals(rewards, qvals_next_max, gamma, n_step)
    print(qvals_esti)

    rewards = [1,]
    qvals_next_max = [2]
    gamma = 0.5
    n_step = 4
    qvals_esti = get_n_step_estimated_qvals(rewards, qvals_next_max, gamma, n_step)
    print(qvals_esti)

    print("Done")


"""
python model.py
"""
