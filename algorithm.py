import argparse
import os
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from AoI_Energy import AoI_Energy

# ATTENTION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

#  ################### params ###########################
parser = argparse.ArgumentParser(description='Hyper_params')
parser.add_argument('--Info', default='', type=str)  # information added to log dir name

parser.add_argument('--Seed', default=0, type=int)
# ATTENTION
parser.add_argument('--slot_Dd', default=1, type=int)

parser.add_argument('--Units', default=256, type=int)  # hidden units num of NN
parser.add_argument('--Lr', default=0.0005, type=float)  # learning rate
parser.add_argument('--Lr_Decay', default=1e-5, type=float)
parser.add_argument('--R_Beta', default=0.0005, type=int)  # learning rate for average reward default=0.0005
parser.add_argument('--Max_Epsilon', default=1.0, type=float)
parser.add_argument('--Min_Epsilon', default=0.01, type=float)
parser.add_argument('--Epsilon_Decay', default=1.0, type=float)
parser.add_argument('--Batch_Size', default=64, type=int)
parser.add_argument('--Memory_Size', default=200000, type=int)  # buffer size
parser.add_argument('--Start_Size', default=50000, type=int)  # step to begin train
parser.add_argument('--Update_Lazy_Step', default=2000, type=int)  # frequency of target update

parser.add_argument('--Evaluate_Interval', default=2000, type=int)  # how often to evaluate (in steps)
parser.add_argument('--Test_Epsilon', default=0.05, type=float)  # adopted epsilon during evaluation
parser.add_argument('--Points', default=150, type=int)  # total evaluation times
parser.add_argument('--Test_Step', default=10000, type=int)  # step number of one evaluation

parser.add_argument('--Alg', default='due', type=str)
parser.add_argument('--Gpu_Id', default='-1', type=str)  # -1 means CPU

parser.add_argument('--User', default=6 0, type=int)  # user number
parser.add_argument('--Beta2', default=0.5, type=float)  # B2 ATTENTION 1
parser.add_argument('--Request_P', default=0.7, type=float)  # request prob
# parser.add_argument('--prioritized', default=False, type=bool)
# ATTENTION
parser.add_argument('--random', default=False, type=bool)
parser.add_argument('--max', default=False, type=bool)

# parser.add_argument('--Gamma', default=0.95, type=float)

args = parser.parse_args()

#  ################### seed ###########################
os.environ['TF_DETERMINISTIC_OPS'] = 'True'
os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"] = 'True'
os.environ["CUDA_VISIBLE_DEVICES"] = args.Gpu_Id
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.random.set_seed(args.Seed)
np.random.seed(args.Seed)

#  ################### log ###########################
# create log file
time_str = time.strftime("%m-%d_%H-%M", time.localtime())
alg = args.Alg
log_dir_name = time_str + '_' + alg + args.Info + '_n' + str(args.User) + '_seed' + str(
    args.Seed) + '_γ1=0.6' + '_batch' + str(args.Batch_Size) + '_slotDd' + str(args.slot_Dd) + '_drn'
fw = tf.summary.create_file_writer(log_dir_name)  # log file writer

# create dir to save model
if not os.path.exists(log_dir_name + '/models'):
    os.makedirs(log_dir_name + '/models')

# save params to a .txt file
prams_file = open(log_dir_name + '/prams_table.txt', 'w')
prams_file.writelines(f'{i:50} {v}\n' for i, v in args.__dict__.items())
prams_file.close()

#  ##################### env ###############################
env = AoI_Energy(user_num=args.User, beta2=args.Beta2, seed=args.Seed, request_p=args.Request_P,
                 # ATTENTION
                 slot_Dd=args.slot_Dd)
Action_Num = env.Actual_Action_Num  # 动作数量386
print("Action_Num:", Action_Num)
Initial_R = - env.C1  # 最大开销 -152.0
print("Inintial_R:", Initial_R)

#  ##################### others ###############################
Optimizer = tf.optimizers.Adam(args.Lr, decay=args.Lr_Decay)
W_Initializer = tf.initializers.he_normal(args.Seed)  # NN initializer
Epsilon_Decay_Rate = (args.Min_Epsilon - args.Max_Epsilon) / args.Memory_Size * args.Epsilon_Decay  # factor of decay
# 衰减因子=(min ε-max ε)/缓冲区大小200000*1
TENSOR_FLOAT_TYPE = tf.dtypes.float32
TENSOR_INT_TYPE = tf.dtypes.int32


# wangc 原本的经验池
class OldReplayBuffer:
    def __init__(self, size):
        self.cap = size
        buffer_s_dim = (size, env.N + 1, env.K)

        self.s_buffer = np.empty(buffer_s_dim, dtype=np.float32)
        self.a_buffer = np.random.randint(0, Action_Num, (self.cap, 1), dtype=np.int32)
        self.r_buffer = np.empty((self.cap, 1), dtype=np.float32)
        self.next_s_buffer = np.empty(buffer_s_dim, dtype=np.float32)

        self.cap_index = 0
        self.size = 0

    def store(self, step):
        s, a, r, next_s = step
        self.s_buffer[self.cap_index] = s
        self.a_buffer[self.cap_index][0] = a
        self.r_buffer[self.cap_index][0] = r
        self.next_s_buffer[self.cap_index] = next_s

        self.cap_index = (self.cap_index + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)

        batch_s = self.s_buffer[idx]
        batch_a = self.a_buffer[idx]
        batch_r = self.r_buffer[idx]
        batch_next_s = self.next_s_buffer[idx]

        return batch_s, batch_a, batch_r, batch_next_s

    def size(self):
        return self.size

# wangc SumTree的实现
class SumTree:
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=float)
        self.data = np.zeros(capacity, dtype=object)

    # 将一条经验的优先级p和数据data存储到SumTree中
    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    # 更新指定节点tree_idx处的优先级值
    def update(self, tree_idx, p):
        # print("tree_idx:",tree_idx)
        change = p - self.tree[tree_idx]
        # print("change:",change)
        # p_scalar = p.item()
        self.tree[tree_idx] = p
        self._propagate(tree_idx, change)

    # 递归地向上更新树的父节点的优先级。
    def _propagate(self, tree_idx, change):
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # 获取叶子节点的索引、优先级和对应的数据
    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    v -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]  # 返回叶子节点的索引、优先级和对应的数据

    # 返回整个 SumTree 的总优先级和
    def total(self):
        return self.tree[0]


# wangc 优先级经验池
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, abs_err_upper=1.):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01
        self.abs_err_upper = abs_err_upper  # 添加 abs_err_upper, 限制误差的上限

        self.tree = SumTree(capacity)
        # self.current_size = 0  # 记录当前存储的经验数量

    def store(self, transition):
        # 检查经验池是否已满，如果已满，则替换最旧的经验
        # if self.current_size < self.capacity:
        #     max_p = np.max(self.tree.tree[-self.capacity:])
        # else:
        #     max_p = np.max(self.tree.tree[-self.capacity: -self.capacity + self.current_size])

        # if self.current_size < self.capacity:
        #     max_p = np.max(self.tree.tree[-self.capacity:])
        # else:
        #     if self.current_size > 0:
        #         max_p = np.max(self.tree.tree[-self.capacity: -self.capacity + self.current_size])
        #     else:
        #         max_p = 1.0  # 或者适当的默认值

        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

        # 更新当前存储的经验数量
        # self.current_size = min(self.current_size + 1, self.capacity)

    def sample(self, batch_size):
        b_idx = []
        b_memory = []
        ISWeights = []

        total_priority = self.tree.total()
        pri_seg = total_priority / batch_size

        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / total_priority
        if min_prob == 0:
            min_prob = 0.00001
        for i in range(batch_size):
            a, b = pri_seg * i, pri_seg * (i + 1)
            sample_value = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(sample_value)
            prob = p / total_priority
            ISWeight = np.power(prob / min_prob, -self.beta)
            b_idx.append(idx)
            b_memory.append(data)
            ISWeights.append(ISWeight)

        return b_idx, b_memory, ISWeights

    # 批量更新存储在经验池中的经验的优先级
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon
        # print("abs_errors:",abs_errors)
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # print("clipped_errors:",clipped_errors)
        ps = np.power(clipped_errors, self.alpha)
        # print("ps:",ps)
        for ti, p in zip(tree_idx, ps):
            # print("tree_idx:",tree_idx)
            # print("ti:",ti)
            # print("p:",p)
            # p_scalar = p.item()
            self.tree.update(ti, p)


# wangc 自定义noisy层
class NoisyLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation='linear', sigma_init=0.5, **kwargs):
        super(NoisyLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.sigma_init = sigma_init

    def build(self, input_shape):
        self.mu_weight = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
            name='mu_weight'
        )

        self.sigma_weight = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True,
            name='sigma_weight'
        )

        self.mu_bias = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='mu_bias'
        )

        self.sigma_bias = self.add_weight(
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(self.sigma_init),
            trainable=True,
            name='sigma_bias'
        )

    def call(self, inputs, noise=True):
        e_w, e_b = self.get_noise_params(noise)
        noisy_weights = self.mu_weight + self.sigma_weight * e_w
        noisy_bias = self.mu_bias + self.sigma_bias * e_b
        outputs = tf.matmul(inputs, noisy_weights) + noisy_bias
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_noise_params(self, noise=True):
        if noise is True:
            e_i = tf.keras.backend.random_normal(shape=tf.shape(self.mu_weight), mean=0.0, stddev=1.0)
            e_j = tf.keras.backend.random_normal(shape=tf.shape(self.mu_bias), mean=0.0, stddev=1.0)
            e_w = tf.keras.backend.sign(e_i) * tf.keras.backend.sqrt(tf.keras.backend.abs(e_i)) * \
                  tf.keras.backend.sign(e_j) * tf.keras.backend.sqrt(tf.keras.backend.abs(e_j))
            e_b = tf.keras.backend.sign(e_j) * tf.keras.backend.sqrt(tf.keras.backend.abs(e_j))
            return e_w, e_b
        else:
            return 0, 0

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)


class drn_agent:
    def __init__(self, max_epsilon, batch_size, memory_size):
        # wangc 向dueling 中加入noisy
        def build_noisy_dueling_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)

            # v(s)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(v_dense)
            v_out = keras.layers.Dense(1, kernel_initializer=W_Initializer)(v_dense)

            # advantages with Noisy Layer
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(
                adv_dense)
            adv_out = NoisyLayer(Action_Num)(adv_dense)
            adv_normal = keras.layers.Lambda(lambda x1: x1 - tf.reduce_mean(x1))(adv_out)

            # q
            outputs = keras.layers.add([v_out, adv_normal])
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        def build_dueling_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)

            # v(s)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            v_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(v_dense)
            v_out = keras.layers.Dense(1, kernel_initializer=W_Initializer)(v_dense)

            # advantages
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(x)
            adv_dense = keras.layers.Dense(args.Units / 2, activation='relu', kernel_initializer=W_Initializer)(
                adv_dense)
            adv_out = keras.layers.Dense(Action_Num, kernel_initializer=W_Initializer)(adv_dense)
            adv_normal = keras.layers.Lambda(lambda x1: x1 - tf.reduce_mean(x1))(adv_out)

            # q
            outputs = keras.layers.add([v_out, adv_normal])
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        # wangc 向DQN中添加noisy
        def build_noisy_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)

            # 添加噪声层
            # x = NoisyLayer(args.Units, activation='relu', sigma_init=0.5)(x)
            # x = NoisyLayer(args.Units, activation='relu', sigma_init=0.5)(x)
            #
            # outputs = keras.layers.Dense(Action_Num, kernel_initializer=W_Initializer)(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)

            # Adding NoisyLayer for the final output layer
            outputs = NoisyLayer(Action_Num, kernel_initializer=W_Initializer)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        def build_net():
            inputs = keras.Input(shape=(env.N + 1, env.K))
            x = keras.layers.Flatten()(inputs)

            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)
            x = keras.layers.Dense(args.Units, activation='relu', kernel_initializer=W_Initializer)(x)

            outputs = keras.layers.Dense(Action_Num, kernel_initializer=W_Initializer)(x)
            model = keras.Model(inputs=inputs, outputs=outputs)
            return model

        if 'due' in alg:  # dueling
            self.active_qnet = build_noisy_dueling_net()  # 评估网络
            self.lazy_qnet = build_noisy_dueling_net()  # target q 目标网络
            print("dueling net")
        elif 'dqn' in alg:  # dqn
            self.active_qnet = build_net()
            self.lazy_qnet = build_net()
            print("dqn net")
        else:
            raise NotImplementedError("alg not implemented")

        self.active_qnet.compile(optimizer=Optimizer, loss='mse')
        self.epsilon = max_epsilon
        self.batch_size = batch_size
        # self.buffer = OldReplayBuffer(memory_size)
        self.buffer = PrioritizedReplayBuffer(memory_size)  # wangc  用于优先经验池的缓冲区
        self.alg = alg
        self.R = Initial_R  # ATTENTION average reward   平均奖励
        # self.gamma = args.Gamma  # wangc discount factor 折扣系数(如果不用R学习可以打开)

    def choose_action(self, s, epsilon):
        # ATTENTION
        if args.random is True:
            # print("=========random==========")
            return np.random.choice(Action_Num)
        elif args.max is True:
            # print("=========greedy==========")
            return tf.argmax(self.active_qnet(s[None, :]), 1)[0].numpy()

        else:
            # print("=========normal==========")
            # if np.random.random() < epsilon:
            #     return np.random.choice(Action_Num)
            # else:
            #     return tf.argmax(self.active_qnet(s[None, :]), 1)[0].numpy()
            # wangc 添加了噪声，不需要使用epsilon-greedy 策略
            return tf.argmax(self.active_qnet(s[None, :]), 1)[0].numpy()

    # wangc 原来的train
    def train(self, batch_size=args.Batch_Size):
        # sample from buffer
        s, a, r, s_next = self.buffer.sample(batch_size)

        # calculate target q
        q_next_lazy = self.lazy_qnet(s_next)
        max_q_next_lazy = tf.reduce_max(q_next_lazy, 1, True)
        # ATTENTION 和DQN不同  q_target = r + self.gamma * max_q_next_lazy  # 乘以折扣系数
        q_target = r - self.R + max_q_next_lazy

        # calculate loss
        with tf.GradientTape() as tape:
            q_active = self.active_qnet(s)
            q_chosen_active = tf.gather(q_active, a, batch_dims=-1)

            td = q_target - q_chosen_active
            loss = tf.reduce_mean(tf.square(td))

        # update R
        self.R += args.R_Beta * tf.reduce_sum(td).numpy()  # wangc dqn不用
        grads = tape.gradient(loss, self.active_qnet.trainable_variables)  # gradients
        grads = [tf.clip_by_norm(grad, 10.0) for grad in grads]

        self.active_qnet.optimizer.apply_gradients(zip(grads, self.active_qnet.trainable_variables))

    # wangc 优先级经验池train
    def train_priorities(self, batch_size=args.Batch_Size):
        # sample from buffer
        b_idx, b_memory, ISWeights = self.buffer.sample(batch_size)
        s, a, r, s_next = zip(*b_memory)  # 解包

        s = np.array(s)
        # print("s:", s)
        a = np.array(a)
        # print("a:", a)
        r = np.array(r)
        # print("r:", r)
        s_next = np.array(s_next)
        # print("s_next:", s_next)

        # calculate target q
        q_next_lazy = self.lazy_qnet(s_next)  # 二维数组
        # print("q_next_lazy:",q_next_lazy)
        max_q_next_lazy = tf.reduce_max(q_next_lazy, 1, True)  # 最大Q值，二维数组
        # print("max_q_next_lazy:",max_q_next_lazy)
        max_q_next_lazy = tf.squeeze(max_q_next_lazy, axis=1)  # 一维数组
        # print("max_q_next_lazy1:",max_q_next_lazy)
        q_target = r - self.R + max_q_next_lazy
        # print("q_target:",q_target)

        # 调试语句
        # print("Min action index:", np.min(a))
        # print("Max action index:", np.max(a))

        # calculate loss
        with tf.GradientTape() as tape:
            q_active = self.active_qnet(s)  # 二维数组
            # print("q_active:",q_active)
            # a = tf.clip_by_value(a, 0, 386)  # 限制a的范围在[0, 386]
            # q_chosen_active = tf.gather(q_active, a, batch_dims=-1)
            q_chosen_active = tf.reduce_sum(q_active * tf.one_hot(a, Action_Num), axis=1)  # 按动作a选择Q值 一维数组
            # q_chosen_active = tf.gather(q_active, a, batch_dims=-1)  # wangc 另一种 二维数组
            # print("q_chosen_active:",q_chosen_active)

            td = q_target - q_chosen_active
            # print("td:",td)
            loss = tf.reduce_mean(ISWeights * tf.square(td))  # Apply importance sampling weights

        # update R
        self.R += args.R_Beta * tf.reduce_sum(td).numpy()
        grads = tape.gradient(loss, self.active_qnet.trainable_variables)
        grads = [tf.clip_by_norm(grad, 10.0) for grad in grads]

        self.active_qnet.optimizer.apply_gradients(zip(grads, self.active_qnet.trainable_variables))
        # 更新优先级
        abs_td = np.abs(td)
        # print("abs_td:",abs_td)
        self.buffer.batch_update(b_idx, abs_td)

    def update_lazy_q(self):
        # update target q
        for lazy, active in zip(self.lazy_qnet.trainable_variables, self.active_qnet.trainable_variables):
            lazy.assign(active)

    def save_model(self, dir_=log_dir_name + '/models'):
        self.lazy_qnet.save_weights(dir_ + '/' + self.alg + '_lazy_qnet.h5')
        self.active_qnet.save_weights(dir_ + '/' + self.alg + '_active_qnet.h5')


def train(points=args.Points):  # points == 150
    agent = drn_agent(args.Max_Epsilon, args.Batch_Size, args.Memory_Size)
    print("============" + agent.alg + "============")

    st = env.reset() / env.AoI_Max  # 初始化状态St
    step = 0  # 每一轮次的步数
    summary_step = 0  # 迭代次数

    while summary_step < points:
        # ATTENTION
        env.update_para()

        env.simulation_user_request()

        a = agent.choose_action(st, agent.epsilon)
        stp1, r = env.step(a)  # St+1, reward
        stp1 = stp1 / env.AoI_Max  # normalize state 归一化
        agent.buffer.store((st, a, r[0], stp1))
        # wangc 更新经验
        # q_lazy = agent.lazy_qnet(st)
        # max_q_lazy = tf.reduce_max(q_lazy, 1, True)
        # q_target = r - agent.R + max_q_lazy
        # q_active = agent .active_qnet(st)
        # q_chosen_active = tf.gather(q_active, a, batch_dims=-1)
        # td = q_target - q_chosen_active
        # abs_td = np.abs(td)
        # agent.buffer.batch_update(b_idx, abs_td)
        # transition = np.hstack((st, a, r[0], stp1))
        # agent.buffer.store(transition)  # have high priority for newly arrived transition
        st = stp1

        # 原来的 train
        # if step > args.Start_Size:  # 50000
        #     agent.train(args.Batch_Size)

        # wangc 使用优先级经验池进行更新
        if step > args.Start_Size:
            # 使用采样数据进行训练
            agent.train_priorities(args.Batch_Size)

        # 原来的 update target q
        # if step % args.Update_Lazy_Step == 0:  # 2000
        #     agent.update_lazy_q()

        # wangc 在训练代理时，使用 优先级经验池进行更新
        if step > args.Start_Size and step % args.Update_Lazy_Step == 0:
            agent.update_lazy_q()

        # evaluate
        if step > args.Start_Size and step % args.Evaluate_Interval == 0:  # Evaluate_Interval = 2000
            feedbacks = evaluate(agent, eps=args.Test_Epsilon)
            greedy_test_mean_r = feedbacks[0]
            aoi_cost = feedbacks[1]
            energy_cost = feedbacks[2]
            # ATTENTION
            extra_cost = feedbacks[3]
            extra_update_times = feedbacks[4]
            print("extra_update_times:", feedbacks[4])

            agent.save_model()

            # log
            with fw.as_default():
                print('\nData stored')
                tf.summary.scalar('greedy005_test_mean_r', greedy_test_mean_r, step=summary_step)
                tf.summary.scalar('R', agent.R, step=summary_step)
                tf.summary.scalar('epsilon', agent.epsilon, step=summary_step)
                tf.summary.scalar('aoi_cost', aoi_cost, step=summary_step)
                tf.summary.scalar('energy_cost', energy_cost, step=summary_step)
                # ATTENTION
                tf.summary.scalar('extra_cost', extra_cost, step=summary_step)
                tf.summary.scalar('extra_update_times', extra_update_times, step=summary_step)

                summary_step += 1
            print("summary_step:",summary_step)

        # epsilon decay
        agent.epsilon = max(Epsilon_Decay_Rate * step + args.Max_Epsilon, args.Min_Epsilon)
        step += 1


def evaluate(agent, n_step=args.Test_Step, eps=0.0):  # n_step = 10000
    test_env = AoI_Energy(user_num=args.User, beta2=args.Beta2, seed=args.Seed, request_p=args.Request_P,
                          # ATTENTION
                          slot_Dd=args.slot_Dd)
    st = test_env.reset() / test_env.AoI_Max
    total_r = 0.0
    aoi_cost = 0.0
    energy_cost = 0.0
    # ATTENTION
    extra_cost = 0.0
    extra_update_times = 0.0

    for _ in range(n_step):
        test_env.simulation_user_request()
        a = agent.choose_action(st, eps)
        stp1, r = test_env.step(a)
        # print("查看奖励", r)
        # total_r=aoi_cost+energy_cost 的相反数
        # 查看奖励(-84.97550000000001, 52.975500000000004, 32.0, 0, 1)
        st = stp1 / test_env.AoI_Max

        total_r += r[0]
        aoi_cost += r[1]
        energy_cost += r[2]
        # ATTENTION
        extra_cost += r[3]
        extra_update_times += r[4]

    return total_r / n_step, aoi_cost / n_step, energy_cost / n_step, extra_cost / n_step, extra_update_times


if __name__ == "__main__":
    train(args.Points)
