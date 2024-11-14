import numpy as np
import cv2 
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
# from multiagent.policy import InteractivePolicy
# import matplotlib.pyplot as plt 


def img2obs(image_array):
    """处理图片获得关键信息,目标位置，智能体位置，追击者位置，障碍物位置(最近的3个障碍物位置)

    Args:
        image_array (nparray): 三通道的bgr图片

    Returns:
        obs (): 目标位置，智能体位置，追击者位置，障碍物位置(最近的3个障碍物位置)
    """
    obstacle_num_in_obs = 3
    pooled_image = cv2.resize(image_array, (800,800), 0, 0, cv2.INTER_MAX)
    _, binary_dst = cv2.threshold(pooled_image[:,:,0], 70, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dst)
    #204,153,204
    goal_lower_bound = np.array([199, 148, 199])
    goal_upper_bound = np.array([209, 158, 209])
    #172,236,172
    agent_lower_bound = np.array([167, 231, 167])
    agent_upper_bound = np.array([177, 241, 177])
    #172,172,236
    adv_lower_bound = np.array([231,165, 165])
    adv_upper_bound = np.array([241,177, 177])
    goal_im = np.array(np.where(np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2))).transpose()[:,:2]
    agent_im = np.array(np.where(np.all((pooled_image>=agent_lower_bound) & (pooled_image<=agent_upper_bound),axis=2))).transpose()[:,:2]
    adv_im = np.array(np.where(np.all((pooled_image>=adv_lower_bound) & (pooled_image<=adv_upper_bound),axis=2))).transpose()[:,:2]
    agent_pos = np.mean(agent_im,axis=0).astype(int)
    agent_pos = np.array((agent_pos[1],agent_pos[0]))

    goal_pos = np.mean(goal_im,axis=0).astype(int)
    goal_pos = np.array((goal_pos[1],goal_pos[0]))
    adv_pos = np.mean(adv_im,axis=0).astype(int)
    adv_pos = np.array((adv_pos[1],adv_pos[0]))

    obstacle_pos = []
    distance = []

    for i in range(num_labels):
        if i!=0:
            obstacle_pos.append(centroids[i].astype(int))
            distance.append(np.linalg.norm(agent_pos-centroids[i]))
    
    #获取distance按大小排序后的index
    sorted_indexes = np.argsort(distance)
    # 绝对位置
    # return np.concatenate((goal_pos,agent_pos,adv_pos,obstacle_pos[sorted_indexes[0]],obstacle_pos[sorted_indexes[1]],obstacle_pos[sorted_indexes[2]]))/256
    # 相对位置
    res = np.concatenate((
    goal_pos - agent_pos,          # 目标位置与代理位置的相对位置
    agent_pos,                     # 代理位置
    adv_pos - agent_pos,           # 敌对代理位置与代理位置的相对位置
    obstacle_pos[sorted_indexes[0]] - agent_pos,  # 第一个障碍物位置与代理位置的相对位置
    obstacle_pos[sorted_indexes[1]] - agent_pos,  # 第二个障碍物位置与代理位置的相对位置
    obstacle_pos[sorted_indexes[2]] - agent_pos   # 第三个障碍物位置与代理位置的相对位置
)) /256
    res_exactpos= np.concatenate((
    goal_pos ,          # 目标位置与代理位置的相对位置
    agent_pos,                     # 代理位置
    adv_pos ,           # 敌对代理位置与代理位置的相对位置
    obstacle_pos[sorted_indexes[0]] ,  
    obstacle_pos[sorted_indexes[1]] ,  
    obstacle_pos[sorted_indexes[2]]    
)) / 256
    return res

def img_to_observation(image):
    template_agent=cv2.imread('./images/agent.png')
    template_adversary=cv2.imread('./images/adversary.png')
    template_check=cv2.imread('./images/check.png')
    template_obstacle=cv2.imread('./images/obstacle.png')
    if template_agent is None or template_adversary is None or template_check is None or template_obstacle is None:
        raise ValueError("Failed to load one or more template images.")
    resized_image = cv2.resize(image, (800, 800), 0, 0, cv2.INTER_MAX)

    def find_object_positions(template):
        result = cv2.matchTemplate(resized_image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.75
        loc = np.where(result >= threshold)
        positions = list(zip(*loc[::-1]))
        return positions
    agent_positions = find_object_positions(template_agent)
    adversary_positions = find_object_positions(template_adversary)
    check_positions = find_object_positions(template_check)
    obstacle_positions = find_object_positions(template_obstacle)
    if agent_positions is None:
        agent_positions = np.array([400,400])
    if adversary_positions is None:
        adversary_positions = np.array([400,400])
    def calculate_mean_position(positions):
        if len(positions) == 0:
            return None
        positions = np.array(positions)
        mean_position = np.mean(positions, axis=0).astype(int)
        return mean_position
    agent_pos = calculate_mean_position(agent_positions)
    adversary_pos = calculate_mean_position(adversary_positions)
    check_pos = calculate_mean_position(check_positions)
    obstacle_pos = []
    distance = []
    for pos in obstacle_positions:
        obstacle_pos.append(pos)
        distance.append(np.linalg.norm(agent_pos - pos))
    sorted_indexes = np.argsort(distance)
    res = np.concatenate((
        check_pos - agent_pos,          # 目标位置与代理位置的相对位置
        agent_pos,                      # 代理位置
        adversary_pos - agent_pos,      # 敌对代理位置与代理位置的相对位置
        obstacle_pos[sorted_indexes[0]] - agent_pos,  # 第一个障碍物位置与代理位置的相对位置
        obstacle_pos[sorted_indexes[1]] - agent_pos,  # 第二个障碍物位置与代理位置的相对位置
        obstacle_pos[sorted_indexes[2]] - agent_pos   # 第三个障碍物位置与代理位置的相对位置
    )) / 256
    res_exactpos = np.concatenate((
    check_pos ,          # 目标位置与代理位置的相对位置
    agent_pos,                     # 代理位置
    adversary_pos ,           # 敌对代理位置与代理位置的相对位置
    obstacle_pos[sorted_indexes[0]] ,  
    obstacle_pos[sorted_indexes[1]] ,  
    obstacle_pos[sorted_indexes[2]]    
)) / 256
    return res


#204,153,204 this is goal
goal_lower_bound = np.array([199, 148, 199])
goal_upper_bound = np.array([209, 158, 209])

#172,236,172 this is agent
agent_lower_bound = np.array([167, 231, 167])
agent_upper_bound = np.array([177, 241, 177])

#172,172,236 this is adversary
adv_lower_bound = np.array([231,165, 165])
adv_upper_bound = np.array([241,177, 177])


if __name__ == '__main__':
    # parse arguments
    scenario = scenarios.load("simple_tag.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, done_callback=scenario.is_done, shared_viewer = True)
    # env.reset()
    obs_pos = [[-0.35, 0.35], [0.35, 0.35], [0, -0.35]] # test point
    env.reset(np.array([[0.0, 0.0], [0.7, 0.7]]), [-0.5, -0.5], obs_pos)
    image = env.render("rgb_array")  
    step = 0
    total_step = 0
    max_step = 400
    image_array = env.render("rgb_array")[0]  
    # cv2.imwrite('./image.png', image_array)
    pooled_image = cv2.resize(image_array, (800,800), 0, 0, cv2.INTER_MAX)
    
    _, binary_dst = cv2.threshold(pooled_image[:,:,0], 70, 255, cv2.THRESH_BINARY_INV)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_dst)

    cv2.imshow('Pooled Image', pooled_image)
    # cv2.imshow('image_arrayarray', image_array)
    # cv2.imshow('THRESH_BINARY', binary_dst)
    goal_im = np.array(np.where(np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2))).transpose()[:,:2]
    agent_im = np.array(np.where(np.all((pooled_image>=agent_lower_bound) & (pooled_image<=agent_upper_bound),axis=2))).transpose()[:,:2]
    adv_im = np.array(np.where(np.all((pooled_image>=adv_lower_bound) & (pooled_image<=adv_upper_bound),axis=2))).transpose()[:,:2]


    cv2.imshow('goal_im', pooled_image[:,:,0] * np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2))
    cv2.imshow('agent_im', pooled_image[:,:,0] * np.all((pooled_image>=agent_lower_bound) & (pooled_image<=agent_upper_bound),axis=2))
    cv2.imshow('adv_im', pooled_image[:,:,0] * np.all((pooled_image>=adv_lower_bound) & (pooled_image<=adv_upper_bound),axis=2))
    
    labels= np.expand_dims(binary_dst,axis=2).repeat(3,axis=2).astype(np.uint8)
    for st in stats[1:]:
        cv2.rectangle(labels, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), (0, 255, 0), 1)
    cv2.imshow('labels', labels)
    contours,hierarchy = cv2.findContours(pooled_image[:,:,0] * np.all((pooled_image>=goal_lower_bound) & (pooled_image<=goal_upper_bound),axis=2),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #寻找轮廓点
    for obj in contours:
        area = cv2.contourArea(obj)  
        #cv2.drawContours(image_array, obj, -1, (255, 0, 0), 4)  
        perimeter = cv2.arcLength(obj,True) 
        approx = cv2.approxPolyDP(obj,0.02*perimeter,True)  
        CornerNum = len(approx)   
        x, y, w, h = cv2.boundingRect(approx)  
        #轮廓对象分类
        if CornerNum ==3: objType ="triangle"
        elif CornerNum == 4:
            if w==h: objType= "Square"
            else:objType="Rectangle"
            cv2.rectangle(pooled_image,(x,y),(x+w,y+h),(0,0,255),5) 
            # cv2.putText(image_array,objType,(x+(w//2),y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.6,(0,0,0),1) 
        elif CornerNum>4: objType= "Circle"
        else:objType="N"
    cv2.imshow('rectangle', pooled_image)
    cv2.waitKey(0)
    agent_pos = np.mean(agent_im,axis=0).astype(int)
    agent_pos = np.array((agent_pos[1],agent_pos[0]))

    goal_pos = np.mean(goal_im,axis=0).astype(int)
    goal_pos = np.array((goal_pos[1],goal_pos[0]))
    adv_pos = np.mean(adv_im,axis=0).astype(int)
    adv_pos = np.array((adv_pos[1],adv_pos[0]))

    obstacle_pos = []
    distance = []

    for i in range(num_labels):
        if i!=0:
            obstacle_pos.append(centroids[i].astype(int))
            distance.append(np.linalg.norm(agent_pos-centroids[i]))
    
    #获取distance按大小排序后的index
    sorted_indexes = np.argsort(distance)
    res=np.concatenate((goal_pos-agent_pos,agent_pos,adv_pos-agent_pos,obstacle_pos[sorted_indexes[0]]-agent_pos,obstacle_pos[sorted_indexes[1]]-agent_pos,obstacle_pos[sorted_indexes[2]]-agent_pos))/256
    print(agent_pos,goal_pos,adv_pos,obstacle_pos)
    print(res)