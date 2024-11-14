import numpy as np
import cv2 
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
# from multiagent.policy import InteractivePolicy
import matplotlib.pyplot as plt 
from image import img2obs
from image import img_to_observation

if __name__ == '__main__':
    # parse arguments
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=scenario.is_success, done_callback=scenario.is_done, shared_viewer = True)
    # env.reset()
    obs_pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]] # test point
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]), [-0.5, -0.5], obs_pos)
    image = env.render("rgb_array")  # 使用这种方法读取图片
    step = 0
    total_step = 0
    max_step = 400
    '''
    # as said, the agents are [adversary , good agent] 
        # in env.agents <array like>  at enviorment.py
        # for each agent:
        # act_n are as seen below
        # [noop, move right, move left, move up, move down]
        # reward_n contains [adversary reward, good agent reward]
        # below are not useful :
        # done_n contains [adversary done, good agent done]
        # info_n contains [adversary info, good agent info]
        # obs_n contains [ [16X1 Array] , [16X1 Array] ] # 16X1 Array is the observation of the agent
    '''
    
    while True:
        act_n =np.array([0, 0, 0, 0, 0]) # not move
        # below are some examples of act_n_random
        act_n_uniform = np.random.rand(5)
        act_n_normal = np.random.normal(loc=0, scale=1, size=5)
        act_n_uniform_range = np.random.uniform(low=0, high=1, size=5)
        act_n_poisson = np.random.poisson(lam=1, size=5)
        act_n_binomial = np.random.binomial(n=10, p=0.5, size=5)
        # you can read obs from the env.step , but u can also read it from the image , plz see image.py 's img2obs functio
        next_obs_n, reward_n, done_n, info_n = env.step(act_n_normal)
        reward_n = np.array([0.0, 0.0])
        # print(f"reward: {reward_n}")
        # print(f"obs: {next_obs_n[0]} \n {next_obs_n[1]}")
        # print(f"vel: {np.linalg.norm(next_obs_n[0][-4:-2])}, vel2: {np.linalg.norm(next_obs_n[0][-2:])}") # vel1: 0.28, vel2: 0.25
        # print(f"done: {done_n}")
        # print(f"info: {info_n}")
        image_ = env.render("rgb_array")[0]
        imgobs= img2obs(image_)
        # print(f"imgobs: {imgobs}")
        imgtoobs= img_to_observation(image_)
        # print(f"imgtoobs: {imgtoobs}")
        error1=imgobs-imgtoobs
        print(f"error1: {error1}")
        # print(f"shape: {np.shape(image)}")
        # print(env._get_info(agent=agent))
        step += 1
        # if step % 50:
        #     print(image)
        #     plt.imshow(image)
        #     plt.show()
        #     # time.sleep(0.0167) # 60 fps
        #     plt.close()

        if True in done_n or step > max_step:
            break
