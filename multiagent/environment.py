import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import random
from multiagent.multi_discrete import MultiDiscrete
# environment for all agents in the multiagent world
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward, hasattr() 函数用于判断对象是否包含对应的属性。
        # self.shared_reward = False  # gai
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.count = 0
        
        

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)#加了数据类型
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    # act_space = MultiDiscrete([[0,act_space.n-1] for act_space in total_action_space])
                    act_space = MultiDiscrete([[0, act_space.n-1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-1, high=+1, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    # ###
    # def _seed(self, seed=None):
    #     if seed is None:
    #         np.random.seed(1)
    #     else:
    #         np.random.seed(seed)

    ### gai
    def step(self, action_n, target=None):
    # def step(self, action_n):
        action_n = np.array((np.zeros(5), action_n))
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        dp = [0, 0, 0]
        distl = [0, 0, 0]
        self.agents = self.world.policy_agents
        land = self.world.landmarks
        # set action for each agent
        # print(action_n)

        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            info_n['n'].append(self._get_info(agent))
            done_n.append(self._get_done(agent))

        # # 追捕者adversary
        delta_pos_a = [obs_n[0][2], obs_n[0][3]]  # 相对位置
        distance_a = np.sqrt(np.sum(np.square(delta_pos_a)))
        d_t = delta_pos_a / distance_a  # the unitary relative-positional vector
        action_n[0][1] = d_t[0]
        action_n[0][3] = d_t[1]
        action_n[0][0], action_n[0][2], action_n[0][4] = 0, 0, 0
        
        # pursuer action
        self._set_action(action_n[0], self.agents[0], self.action_space[0]) 
        # evader action
        # self._set_action(action_n[1], self.agents[1], self.action_space[1]) 
        escape_direction = self.calculate_escape_direction(self.agents[1], self.world)
        action_n[1][1] = escape_direction[0]
        action_n[1][3] = escape_direction[1]
        action_n[1][0], action_n[1][2], action_n[1][4] = 0, 0, 0
        self._set_action(action_n[1], self.agents[1], self.action_space[1]) 
        
        # # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    # should be used without obstacles , origin from source code 
    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n
    
    # just updated from _reset , add the obstacles
    def reset(self, agent_pos = None, check_pos = None, obstacles=None):
        # reset world
        self.reset_callback(self.world, agent_pos, check_pos, obstacles)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n    

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        # return self.observation_callback(agent, self.world)
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)


    def get_nearest_adv(self,agent,world):
        nearest_adv = None
        min_dist = 100000
        for a in agent:
            if not a.adverary:
                good_agent = a
        for a in agent:
            if a.adverary:
                dist = np.linalg.norm(good_agent.state.p_pos - a.state.p_pos)
                if dist < min_dist:
                    min_dist = dist
                    nearest_adv = a
        return nearest_adv
    
    def get_nearest_obstacle(self, agent, world):
        nearest_obstacle = None
        min_dist = 100000
        for i, landmark in enumerate(world.landmarks):
            dist = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_obstacle = landmark
        return nearest_obstacle
    
    def get_check_point(self,agent,world):
        for check in self.checkpoints(world):
            dist = np.linalg.norm(agent.state.p_pos - check.state.p_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_check = check
        return nearest_check
    
    def calculate_escape_direction(self, agent, world):
        # 获取最近的敌对智能体
        nearest_adv = world.agents[0]
        escape_direction = agent.state.p_pos - nearest_adv.state.p_pos
        distance_to_adv = np.linalg.norm(escape_direction)
        escape_direction /= distance_to_adv
        # 预测捕食者的下一个位置
        # 预测捕食者的移动方向
        adv_velocity = nearest_adv.state.p_vel
        adv_direction = adv_velocity / np.linalg.norm(adv_velocity)
        
        # 获取目标点
        goal = world.check[0]
        goal_direction = goal.state.p_pos - agent.state.p_pos
        goal_direction /= np.linalg.norm(goal_direction)
    
        # 获取最近的障碍物
        nearest_obs = self.get_nearest_obstacle(agent, world)
        obs_direction = agent.state.p_pos - nearest_obs.state.p_pos
        obs_distance = np.linalg.norm(agent.state.p_pos - nearest_obs.state.p_pos)
        obs_direction /= obs_distance
    
        # 设置阈值
        threshold = 0.28
    
        # 计算最终方向
        if distance_to_adv < threshold:
            # 在远离敌对智能体的同时，朝着目标位置移动
            escape_weight = 0.9 - 0.8 * (distance_to_adv / threshold)
            goal_weight = 1.0 - escape_weight
            final_direction = escape_weight * escape_direction + goal_weight * goal_direction
        else:
            final_direction = 0.8 * goal_direction


        # 增加对障碍物的避让策略
        if obs_distance < 0.16:  # 如果距离障碍物很近，增加避让权重
            final_direction = +0.8 * obs_direction - 0.2 * goal_direction
            
         # 引入随机扰动
        if distance_to_adv > threshold and distance_to_adv < threshold+0.2:  # 在threshold附近引入随机扰动
            perturbation = np.random.normal(0, 0.5, 2)
            final_direction += perturbation
        final_direction /= np.linalg.norm(final_direction)
        
        # 考虑捕食者的移动方向
        # if np.dot(final_direction, adv_direction) > 0:  # 如果最终方向与捕食者的移动方向一致
        #     final_direction -= 0.5 * adv_direction  # 调整方向以避免直接迎头相撞
        return final_direction
 
    def _set_direction(self, agent, direction):
        # Normalize the direction vector
        direction /= np.linalg.norm(direction)
        # Set the agent's action based on the direction
        agent.action.u = direction
        # Apply sensitivity (acceleration)
        sensitivity = 50.0
        if agent.accel is not None:
            sensitivity = agent.accel
        agent.action.u *= sensitivity
        
        
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)  # core.py line 91 定义 self.dim_p=2 ##agent.action.u为2维数据[0,0]

        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]
        # print('a',action)
        if agent.movable:
            # physical action
            # print('ddd:',agent.action.u)
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action  #离散动作空间中 action 为1*5维的向量，action[0][0]为NOOP，即无操作，
                # 其余包括x轴正负向变化量大小，y轴正负向变化量大小
                # 在multi_discrite.py文件里面有说明\
                if action[0] == 1:  agent.action.u[0] = -1.0
                if action[0] == 2:  agent.action.u[0] = +1.0
                if action[0] == 3:  agent.action.u[1] = -1.0
                if action[0] == 4:  agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    # 连续环境的输出
                    agent.action.u = action[0]
                    # print('qqq:', action[0])
                # print("ccc:", agent.action.u)  # 查看信息
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel # 加速度
            agent.action.u *= sensitivity
            # print("action", action)
            action = action[1:]  # action为空列表
            # print("ddd:",agent.action.u) #查看信息
        # make sure we used all elements of action
        assert len(action) == 0


    # reset rendering asset
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(800, 800) # 修改显示框大小

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                elif 'border' in entity.name:
                    geom = rendering.make_polygon(entity.shape)# 画出边界border，形状为方形块
                    #print(entity.shape)
                    geom.set_color(*entity.color)
                    # print("border geom")
                elif 'check' in entity.name:
                    geom = rendering.make_polygon(entity.shape)# 画出check，形状为方形块
                    #print(entity.shape)
                    geom.set_color(*entity.color)
                    # print("border geom")
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            # self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            self.viewers[i].set_bounds(-cam_range, cam_range, -cam_range, cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame局部坐标系
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
