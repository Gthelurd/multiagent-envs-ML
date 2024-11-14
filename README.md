### Multi-Agent Particle Environment Documentation（此版本较为老旧，暂时还没有更新）

#### 简介
本项目的原始代码来源于OpenAI的GitHub仓库：Multi-Agent Particle Environment。在阅读此文档之前，请先阅读此`README.md`文档，里面有对各个代码文件的系统性介绍。

本课程所用的环境为`simple_tag.py`子环境，并进行了一定程度的修改，主要改动为在OpenAI的原始环境基础上，给`simple_tag`环境添加了边界墙、3个固定障碍物和1个目标点。

#### 环境配置(请参考pdf文件)

##### 依赖
根据实操后，需要以下依赖：
- Python 3.6及以上（在我电脑上安装的是Python 3.8，低于Python 3.6可能会报错）
- Gym 0.10.5
- Numpy
- Scipy
- PIL (Pillow)

推荐使用Anaconda创建虚拟环境，在虚拟环境中配置。

安装均可在终端用pip的形式安装：
```bash
pip install gym==0.10.5
pip install numpy
pip install scipy
pip install pillow
```

推荐使用Ubuntu系统，Windows我没试过，不保证能成功运行环境。

##### 安装环境
OpenAI将该环境封装成了库，可以直接通过import形式输出环境。在此之前我们需要安装该环境。

首先切换到环境`multiagent-envs-ML`主目录下（你会看到有一个 `setup.py` 文件），在此目录下打开终端，运行以下代码安装:
```bash
pip install -e .
```
注意 . 号前面有空格。

安装完成后最终会提示：
```bash
Running setup.py develop for multiagent
Successfully installed multiagent-0.0.1
```

#### 代码运行
安装成功后，正常情况下代码运行应该不会报错。运行步骤如下:
```bash
cd multiagent-envs-ML/
python demo.py
```
你会看到环境窗口，同时捕食者和猎物在环境中运动，你也可以看出来这个时候，猎物总是能被捕食者捕获的。

#### 环境代码部分介绍

##### 核心文件
最重要的环境文件是`core.py`，`simple_tag.py`，`environment.py`这三个文件。这三个文件的大致调用关系是：`core`被`simple_tag`调用，`simple_tag`被`environment`调用。

- **core.py**: 主要声明了环境中出现的各个实体：agent, border, landmark, check等。
- **simple_tag.py**: 主要设置实体的参数（初始位置、加速度等），reward设置等。
- **environment.py**: 强化学习算法中的经典环境接口（step, reset, render等）。你可以按照给定任务重写`environment.py`文件中的`self._set_action`函数。

##### 控制量
控制量`action`均为离散量，范围为\([0,1]\)，是个2\*5维的向量，对应两个智能体。每个智能体都是由1\*5维的`action`控制。其中，第1维并没有用到，用到的是后面四维。（官方给的代码默认设置是这样的）

##### 策略和可视化
`policy.py`文件是使用键盘控制写的文件，只是为了演示(`Interactive.py`文件中使用)。

整个画面的实际尺度为\([-1,1]*[-1,1]\)，屏幕正中心为原点。画面宽高为800*800像素。

##### 图像信息提取
在`/multiagent-envs-ML/bin`文件夹下的`interactive.py`文件，有初步获取图像信息的方式`image = env.render("rgb_array")`，至于如何利用这个信息来提取目标信息，可以参考`image.py`。

上一届的同学已经通过这种方式获取图像信息，再通过图像处理的方式(CenterNet算法)获得所有智能体、目标点、障碍物的位置信息。这也是老师所提出的第一个任务要求：通过图像处理获取各个物体的位置信息。

##### 直接信息获取
在`simple_tag.py`文件中有直接获取位置等信息的代码：
```python
def observed(self, agent, world):
    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in world.landmarks:
        if not entity.boundary:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
    # communication of all other agents
    comm = []
    other_pos = []
    other_vel = []
    check_pos = []
    check_pos.append(agent.state.p_pos - world.check[0].state.p_pos)
    for other in world.agents:
        if other is agent:
            continue
        comm.append(other.state.c)
        other_pos.append(other.state.p_pos - agent.state.p_pos)
        other_vel.append(other.state.p_vel)
    dists = np.sqrt(np.sum(np.square(agent.state.p_pos - other_pos)))
    return np.concatenate([agent.state.p_pos] + other_pos + check_pos + entity_pos + [agent.state.p_vel] + dists)
```
但是这种方式是不符合老师所提出的要求的，在这里提个醒（老师是想让同学们用图像处理等知识获取状态信息）。

#### Related resources
- [petting-zoo](https://github.com/Farama-Foundation/petting-zoo)
- [multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)
- [基于TorchRL的竞争性多智能体强化学习 (DDPG) 教程](https://pytorch.ac.cn/rl/0.5/tutorials/multiagent_competitive_ddpg.html)
- [使用 TorchRL 教程进行多智能体强化学习（PPO）](https://pytorch.ac.cn/rl/0.5/tutorials/multiagent_ppo.html)