import random
from itertools import combinations
import numpy as np
from scipy.special import comb


class AoI_Energy(object):

    def __init__(self, seed=0, user_num=20, beta2=0.4, request_p=0.6, slot_Dd=1, **kwargs):
        #ATTENTION 新加
        self.Dd1 = None
        self.user_req_ind = None
        self.Last_State = None

        User_Num = user_num  # number of users 用户数量->20
        Sensor_Num = 10  # number of sensors 传感器数量->10
        Max_Update_Num = 4  # supported maximum number of updated sensors 支持的最大更新传感器数
        Updating_Energy_Cost = np.full((Sensor_Num,), 10, dtype=np.float64)  # Eu (Sensor_Num,) 传感器更新能耗10
        User_Req_Prob_Matrix = np.full((User_Num,), request_p, dtype=np.float64)  # 请求概率 (User_num,)
        Sensor_Popularity = np.full((Sensor_Num,), 1 / Sensor_Num,
                                    dtype=np.float64)  # popularity of request (Sensor_Num,)
        Fail_Prob = np.array([0.02, 0.02, 0.04, 0.04, 0.06, 0.06, 0.08, 0.08, 0.1, 0.1],
                             dtype=np.float64)  # failure probability (Sensor_Num,) 故障概率（传感器数量，）
        # Fail_Prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                      dtype=np.float64)  # failure probability (Sensor_Num,) 故障概率（传感器数量，）
        # Bandwidth = np.array([1, 1.3, 1.7, 2.0])
        Omega_Pro = np.ones(User_Num, dtype=np.float64) * (1 / User_Num)  # 不同用户的权重（一样重要）
        Dd = slot_Dd  # slot of Dd
        # 0.6\0.7\0.8 ---- 0.6
        Beta_1 = 1 - beta2  # AOI开销所占的权重 0.6
        Beta_2 = beta2  # energy开销所占的权重 0.4
        Para = 10  # AOI_max的参数
        Para_mid = 20  # AOI_mid的参数

        self.N = User_Num  # 用户数量
        self.K = Sensor_Num  # 传感器数量 10
        self.M = Max_Update_Num  # 支持的最大更新传感器数 4

        self.Dd = int(Dd)  #
        self.W = Omega_Pro  # 不同用户的权重（根据用户编号）
        self.B1 = Beta_1  # AOI成本的权重（0.6）
        self.B2 = Beta_2  # 能量开销的权重（0.4）
        # ATTENTION
        self.AoI_Max = Para * self.K * (self.Dd )  #
        self.AoI_Mid = Para_mid * self.K * (self.Dd)  #
        self.Fail_Prob_Pro = 1 - np.power((1 - Fail_Prob), self.Dd)  # failure probability 失败概率
        self.E = np.zeros(self.K)
        self.E = Updating_Energy_Cost  #
        self.Cost_Upper_Bound = self.B1 * sum(self.W * self.AoI_Max) + self.B2 * sum(self.E[0:self.M])
        self.C1 = self.Com_Factor * self.Cost_Upper_Bound  # 开销

        self.User_Req_Prob = np.zeros((self.N, self.K + 1))
        self.User_Req_Prob[:, 1:(self.K + 1)] = User_Req_Prob_Matrix.reshape(-1, 1) * Sensor_Popularity # 0.06
        # probability of users' request 用户请求的概率
        self.User_Req_Prob[:, 0] = 1 - np.sum(self.User_Req_Prob[:, 1:self.K + 1],
                                              axis=1)  # 第一列没有请求的概率 (self.N, self.K+1)
        # print("--------------------------用户请求概率-------------------------")
        # print(self.User_Req_Prob)
        self.State_Space = np.ones((self.N + 1, self.K))
        self.Actual_Action_Num = self._possible_action_num(self.K, self.M)
        self.Action_Space = np.zeros((self.Actual_Action_Num, self.K), dtype=np.int64)  # (actual_action_num, K)
        self._action_space_fill()  # fill action space 填补动作空间
        # print("------------------------------------------------动作空间-----------------------------------------------")




    @staticmethod
    # 可能的动作数量（传感器数量10，支持的最大更新传感器数4）
    def _possible_action_num(Q, P):
        A = 0
        for i in range(0, P + 1):  # range(0, 5)
            A += comb(Q, i, exact=True)  # CQ1 + CQ2 + ... + CQP Q选i
        return A

    # 使用0或1填充动作空间
    def _action_space_fill(self):
        """
        fill the action space using 0 or 1 使用0或1填充动作空间
        """
        characters = np.arange(self.K)  # 传感器数量(0,1,...,9)
        indices = []
        for i in np.arange(self.M + 1):  # 支持更新的最大传感器数量(0,1,...,4)
            for item in combinations(characters, i):  # combinations 排列组合
                indices.append(item)

        for j in range(self.Actual_Action_Num):
            self.Action_Space[j, indices[j]] = 1

    # 模拟用户的请求行为，并返回一个表示用户请求的指示器数组。（本实验中一个用户一次只请求一个HDM）
    def _user_request_ind(self):
        # 模拟用户请求行为
        user_req_ind_ = np.zeros(self.N, dtype=np.int64)  # 存储每个用户的请求指示器。
        sensor_seq = np.arange(0, self.K + 1, 1)  # （0，1，2，3，4，5，6，7，8，9，10）
        for i in range(self.N):
            # print(self.User_Req_Prob[i, :], "*" * 50)  # [0.4 0.06 0.06 0.06 ... 0.06]
            user_req_ind_[i] = np.random.choice(sensor_seq, p=self.User_Req_Prob[i, :])  # 从sensor_seq中随机采1个样
        # print(user_req_ind_,"*"*50)
        # [1 3 0 0 2 0 1 9 0 2 0 7 0 0 0 0 1 0 1 0]
        return user_req_ind_  # 返回请求行为--0为没有请求，1-10代表了请求十个中具体哪个传感器

    # 根据系统的状态和请求情况进行更新，计算额外的开销和更新次数，并限制状态空间中AoI值的上边界。
    def _update_state(self):

        update_flag = sum(self.Action_Row) != 0  # 是否进行更新


        self.D = 0
        if update_flag:
            self.D += self.Du

        if (not update_flag) :
            self.D = 1

        # 更新边缘缓存节点的AOI
        # 获取需要更新的请求传感器的索引
        Update_Req = np.where(self.Action_Row == 1)[0]
        self.Update_Result = np.zeros(self.K)
        for i in Update_Req:
            # 根据每个请求传感器的失败概率生成一个随机值
            value = np.random.choice([0, 1], p=[self.Fail_Prob_Pro[i], 1 - self.Fail_Prob_Pro[i]])
            self.Update_Result[i] = value  # 更新结果--0为失败1为成功
        # print(self.Update_Result,"---------------")
        # [0. 0. 0. 1. 0. 1. 0. 0. 0. 1.] ---------------
        # for i in list_update_req:
        #     value = np.random.choice([0, 1], p=[self.Fail_Prob_Pro[i], 1 - self.Fail_Prob_Pro[i]])
        #     self.Update_Result[i] = value  # 更新结果--0为失败1为成功
        # print(self.Update_Result)
        # [0. 0. 0. 1. 0. 1. 0. 1. 0. 1.]

        # 将更新成功的传感器的AoI设置为self.D，将更新失败的传感器的AoI增加self.D
        indices_suc = np.where(self.Update_Result == 1)[0]
        indices_fail = np.where(self.Update_Result == 0)[0]
        self.State_Space[0, indices_suc] = self.D  # 更新成功时隙重置为1
        self.State_Space[0, indices_fail] = self.State_Space[0, indices_fail] + self.D  # 更新的失败时隙加1

        # 更新用户的AOI
        # 将用户的AoI值增加self.D
        # 选择了状态空间中除第一行（表示边缘缓存节点的 AoI 值）之外的所有行（表示用户的AoI值），并选择了所有的列。
        # 这样，我们就得到了一个形状为(N, K)的子数组，其中N是用户数量，K是传感器数量。
        self.State_Space[1:self.N + 1, :] = self.State_Space[1:self.N + 1, :] + self.D
        actual_user_req_ind = np.where(self.user_req_ind != 0)[0]  # 返回实际的用户请求
        # print(actual_user_req_ind,"*"*50)
        # [ 7  8 19] **************************************************
        # [ 4  7  8  9 10 17] **************************************************
        # [2] **************************************************
        # 根据实际用户请求的索引和请求传感器的编号，更新状态空间中对应位置的AoI值
        # 实际请求用户，请求的传感器号 一一对应
        state_space_update_indices = (actual_user_req_ind + 1, self.user_req_ind[actual_user_req_ind] - 1)
        # print(state_space_update_indices,"*"*50)
        # (array([13], dtype=int64), array([0], dtype=int64)) **************************************************
        # (array([ 9, 16], dtype=int64), array([1, 6], dtype=int64)) **************************************************
        # (array([1, 6, 8], dtype=int64), array([9, 9, 2], dtype=int64)) **********************************************
        self.State_Space[state_space_update_indices] = self.State_Space[0, state_space_update_indices[1]]

        # print(self.State_Space[:, :])  # 观察状态空间21(1个缓存节点和20个用户)*10
        over_indices = np.where(self.State_Space[:, :] > self.AoI_Max)  # 找到状态空间中AOI值大于最大限制的状态的位置
        self.State_Space[over_indices] = self.AoI_Max  # 将这些大于最大限制的AOI重置为AOI_Max
        # over_indices的类型：array d_type=int64

        # ud AOI上边界
        # over_indices_mid = np.where(self.State_Space[:, :] > self.AOI_Mid)
        # self.State_Space[over_indices_mid] = self.AOI_Mid

    # 执行动作并更新系统状态，同时计算奖励、开销和下一个状态空间
    def _step(self, action):
        # print("------------------------------当前动作-----------------------------")
        # 358
        # print(action)
        # print(self.Actual_Action_Num)
        # 456
        # 检查给定的动作 action 是否在动作空间中存在.any()有一个为True即为True，否则触发异常
        assert ((np.arange(self.Actual_Action_Num) == action).any()) == True  # .any()有一个为True即为True，否则触发异常
        self.Action_Row = self.Action_Space[action, :]  # 取动作空间中对应action的那一行
        # print("----------------------------------action row------------------------------")
        # Action_Row: [0 0 0 0 1 0 0 1 1 1]
        # print("Action_Row:", self.Action_Row)
        self._update_state()  # 更新缓存和用户状态

        self.AoI_Table = np.zeros((self.N, self.K, self.D + 1))  # 用户的数量，传感器的数量，一般步长的持续时间
    
        # 一种更有效的计算"传感器"的方法
        sensors = (self.Last_State[1:] + 1. + self.State_Space[1:]) / 2.
        sensors = np.minimum(sensors, self.AoI_Max)

        # 计算每个用户的平均 AoI 成本 users，通过对 sensors 的第二个维度求和并除以 self.K 得到。
        users = np.sum(sensors, axis=1) / self.K  # (N,)  users = Δnt 用户的平均AoI成本

        self.AoI_Cost = int(np.sum(self.W * users * 10000)) / 10000
        # print(self.AoI_Cost)
        self.Energy_Cost = int(sum(self.Action_Row * self.E * 10000)) / 10000
        # self.Energy_Cost = 10 * int(sum(self.Action_Row * self.E * 10000)) / (10000 * self.K)

        aoi_cost = self.B1 * self.AoI_Cost
        energy_cost = self.B2 * self.Energy_Cost
        #ATTENTION
        extra_cost = self.Extra_cost
        #ATTENTION         self.Total_Cost = aoi_cost + energy_cost
        self.Total_Cost = aoi_cost + energy_cost   + extra_cost  # + sum(self.Action_Row) / self.K
        # TODO 添加一个资源分配的回报值
        self.reward = - self.Total_Cost  # real reward 实际回报
        self.Last_State = self.State_Space.copy()

        return self.State_Space.copy(), (self.reward, aoi_cost, energy_cost, extra_cost)

    # 返回重置的状态空间
    def reset(self):
        # 初始态
        #ATTENTION self.State_Space = np.zeros((self.N + 1, self.K))
        self.State_Space = np.random.randint(20, 100, size=(self.N + 1, self.K))  # 状态空间初始化

        self.Last_State = self.State_Space.copy()
        return self.State_Space.copy()  # 返回重置的状态空间

    # 模拟用户请求
    def simulation_user_request(self):  # 模拟用户请求
        self.user_req_ind = self._user_request_ind()
        return self.user_req_ind.copy()  # 返回模拟的用户请求

    # 返回 状态空间, (回报, AOI开销, latency开销)
    def step(self, action):
        return self._step(action)  # 返回 状态空间, (回报, AOI开销, 能量开销)


if __name__ == '__main__':
    env = AoI_Energy()
