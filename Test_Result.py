import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.envs.classic_control import utils
from typing import Optional
import importlib
import sys
import matplotlib.pyplot as plt
from collections import deque
import time
import csv

##Set the basic parameters
balance_time = 20  # 平衡任务的总时长(秒)
h_in = 1/100       # 时间步长(0.01秒，即每秒100步)
## balance_time: 定义游戏持续的总时间
## h_in: 控制算法的时间步长，表示模拟精度

def save_episode_results_to_csv(episode_results, current_range, save_path="Episode_State"):
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 创建文件名
    filename = f"episode_results_range_{current_range:.3f}.csv".replace('.', '_')
    full_path = os.path.join(save_path, filename)
    
    try:
        # 写入CSV文件
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入表头
            writer.writerow(['success_label','x', 'x_dot', 'theta', 'theta_dot'])
            
            # 写入数据
            for state_data in episode_results:
                writer.writerow([state_data['success_label'],
                               state_data['x'], 
                               state_data['x_dot'],
                               state_data['theta'], 
                               state_data['theta_dot']])
        
        print(f"✅ Episode结果已保存到: {full_path}")
        return full_path
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return None

# 定义平衡判断函数
def check_balance_status_fixed(x, theta, x_dot, theta_dot):
    """修正后的平衡状态判断函数"""
    if x >= -0.1 and x <= 0.1 and theta > -0.1 and theta < 0.1 and x_dot >= -0.1 and x_dot <= 0.1 and theta_dot >= -0.1 and theta_dot <= 0.1:
        quality = 100
        status = "Perfect"
        is_balanced = True
    else:
        quality = 0
        status = "Lost"
        is_balanced = False
    return quality, status, is_balanced


def create_dynamic_reset_method(new_low, new_high):
    """
    创建新的reset方法，使用指定的范围
    """
    print(f"🔧 创建动态reset方法，范围: [{new_low}, {new_high}]")
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # 调用父类的reset（但不传seed，避免重复设置）
        super(type(self), self).reset(seed=seed)
        
        # 🔧 直接使用传入的范围，不依赖utils.maybe_parse_reset_bounds
        low_bound = np.array([new_low, new_low, new_low, new_low], dtype=np.float32)
        high_bound = np.array([new_high, new_high, new_high, new_high], dtype=np.float32)
        
        # 生成随机初始状态
        self.state = self.np_random.uniform(low=low_bound, high=high_bound, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
            
        # 🔍 调试输出：验证生成的状态
        x, x_dot, theta, theta_dot = self.state
        print(f"    🎲 生成状态: x={x:.3f}, θ={theta:.3f}, x_dot={x_dot:.3f}, θ_dot={theta_dot:.3f}")
        
        return np.array(self.state, dtype=np.float32), {}
    
    return reset

def update_environment_reset_range(new_low, new_high):
    """
    更新环境的reset范围
    """
    print(f"🚀 开始更新环境reset范围: [{new_low}, {new_high}]")
    import myCartpoleF
    import inspect
    # 检查原始reset方法
    original_reset = myCartpoleF.myCartPoleEnvF.reset
    print(f"📋 原始reset方法: {original_reset}")
    
    # 创建新的reset方法
    new_reset_method = create_dynamic_reset_method(new_low, new_high)
    
    # 替换类方法
    myCartpoleF.myCartPoleEnvF.reset = new_reset_method
    
    # 验证替换是否成功
    updated_reset = myCartpoleF.myCartPoleEnvF.reset
    print(f"📋 更新后reset方法: {updated_reset}")
    return None
    
## current_range是更新好的初始范围，这个会在主方法调用的时候修改处理
def run_100_tests_with_dynamic_reset(Test_Path, current_range):
    print(f"\n🚀 开始10000次测试: ±{current_range}")
    print("="*60)
    
    # 🔧 先重载模块（清理之前的修改）
    import importlib
    import myCartpoleF
    importlib.reload(myCartpoleF)
    
    # 🔧 重新应用reset方法修改
    print("🔧 重新应用reset方法...")
    update_success = update_environment_reset_range(-current_range, current_range)
    
    # 📝 注册新环境
    dynamic_env_name = f'CartPoleLab_Dynamic_{current_range:.3f}'.replace('.', '_')
    
    try:
        gym.register(
            id=dynamic_env_name,
            entry_point='myCartpoleF:myCartPoleEnvF',
            reward_threshold=balance_time / h_in * 0.95,
            max_episode_steps=int(balance_time / h_in)
        )
        print(f"✅ 环境注册成功: {dynamic_env_name}")
    except Exception as e:
        print(f"❌ 环境注册失败: {e}")
        return None
    
    # 🎮 创建环境
    print("🎮 创建环境...")
    env = gym.make(dynamic_env_name, render_mode=None)
    env = DummyVecEnv([lambda: env])
    model = PPO.load(Test_Path, env=env)
    
    # 📊 初始化统计
    all_episodes_data = []
    total_test_stats = {
        'total_episodes': 0,
        'successful_episodes': 0,
        'average_steps': 0,
        'average_balance_rate': 0,
        'perfect_episodes': 0,
        'excellent_episodes': 0,
        'good_episodes': 0,
        'failed_episodes': 0
    }
    
     # 专门记录成功/失败标记和初始状态的list
    episode_results = []
    
    print("🎮 开始10000次episode测试...")
    
    ## 是否存在失败情况
    is_false_or = True
    
    ## 这里面填充测试代码
    for i in range(10000):
        obs = env.reset()
        
        # 🔧 解析初始观测值
        if obs.ndim == 2:
            initial_state = obs[0]
        else:
            initial_state = obs
        
        x, x_dot, theta, theta_dot = initial_state
        print(f"Episode {i+1:3d}: 初始状态 x={x:.3f}, θ={theta:.3f}, x_dot={x_dot:.3f}, θ_dot={theta_dot:.3f}")
        
        # 📊 单个episode的统计数据
        episode_stats = {
            'episode_id': i + 1,
            'initial_state': initial_state.copy(),
            'total_steps': 0,
            'perfect_steps': 0,
            'excellent_steps': 0,
            'good_steps': 0,
            'basic_steps': 0,
            'struggling_steps': 0,
            'lost_steps': 0,
            'final_status': '',
            'episode_duration': 0,
            'success': False
        }

        start_time = time.time()
        step_count = 0
        done = False
        
        # 🎮 运行单个episode
        while not done and step_count < 2000:  # 最大2000步
            # 模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step_count += 1
            
            # 解析当前状态
            if obs.ndim == 2:
                current_state = obs[0]
            else:
                current_state = obs
                
            x, x_dot, theta, theta_dot = current_state
            
            # 🎯 评估当前平衡状态
            balance_quality, balance_status, is_balanced = check_balance_status_fixed(x, theta, x_dot, theta_dot)
            
            # 🔧 添加：统计平衡步数
            episode_stats['total_steps'] += 1
            if balance_quality >= 95:
                episode_stats['perfect_steps'] += 1
            elif balance_quality >= 80:
                episode_stats['excellent_steps'] += 1
            elif balance_quality >= 60:
                episode_stats['good_steps'] += 1
            elif balance_quality >= 40:
                episode_stats['basic_steps'] += 1
            elif balance_quality >= 20:
                episode_stats['struggling_steps'] += 1
            else:
                episode_stats['lost_steps'] += 1
        
        # 📊 计算episode结果
        episode_stats['episode_duration'] = time.time() - start_time
        episode_stats['final_status'] = balance_status
        
        # 计算该episode的平衡率
        balanced_steps = (episode_stats['perfect_steps'] + 
                        episode_stats['excellent_steps'] + 
                        episode_stats['good_steps'] + 
                        episode_stats['basic_steps'])
        
        episode_balance_rate = (balanced_steps / episode_stats['total_steps']) * 100 if episode_stats['total_steps'] > 0 else 0
        episode_stats['balance_rate'] = episode_balance_rate
        
        # 🎯 判断episode成功与否
        if episode_balance_rate >= 40 and episode_stats['total_steps'] >= 1000:
            episode_stats['success'] = True
            total_test_stats['successful_episodes'] += 1
            success_label = 1
        else:
            episode_stats['success'] = False
            is_false_or = False
            total_test_stats['failed_episodes'] += 1
            success_label = 0
            
        # 保存为包含五个独立数据的字典
        episode_data = {'success_label': success_label,
            'x': float(initial_state[0]),
            'x_dot': float(initial_state[1]),
            'theta': float(initial_state[2]),
            'theta_dot': float(initial_state[3])
        }
        episode_results.append(episode_data)

        # 💾 保存episode数据
        all_episodes_data.append(episode_stats)
        
        # 🖥️ 实时输出进度
        status_icon = "✅" if episode_stats['success'] else "❌"
        print(f"  {status_icon} 步数:{step_count:4d} | 平衡率:{episode_balance_rate:5.1f}% | 状态:{balance_status}")

    env.close()
    
    # 保存episode_results到文件
    save_episode_results_to_csv(episode_results, current_range, save_path="Episode_State")

    # 🔄 返回结果给主循环使用
    return {
        'average_balance_rate': total_test_stats['average_balance_rate'],
        'average_steps': total_test_stats['average_steps'],
        'if_fail': is_false_or,
        'all_episodes_data': all_episodes_data,
        'total_stats': total_test_stats,
        ## This is the code for success or not, and save the initial state
        'episode_results': episode_results
    }

# 主循环：动态修改reset以及每次循环的10000次测试
def dynamic_reset_loop(Test_path):
    print("开始动态reset范围测试循环")
    
    # 参数设置
    # 验证一下是否成功修改了reset条件
    initial_range = 0.08
    range_step = 0.02
    max_range = 0.20
    current_range = initial_range
    
    ## 这一部分是转换范围的循环范围（这里应该还需要修改一下）
    while current_range <= max_range:
        print(f"\n{'='*60}")
        print(f"当前测试范围: ±{current_range}")
        print(f"{'='*60}")
        # 动态更新reset范围
        update_environment_reset_range(-current_range, current_range)
        # 运行100次测试
        result = run_100_tests_with_dynamic_reset(Test_path, current_range)
        # 添加result的分析逻辑，看看是否符合要求；如果结果不稳定，则退出循环，保留current range
        if result['if_fail'] == False:
            print(f"\n 检测到问题！")
            print(f"   建议的稳定训练范围: ±{current_range:.3f}")
            ## break
        else:
            print(f"\n 范围 ±{current_range} 测试通过")
            print(f" 继续测试更大范围...")
        # 扩大范围（这里应该还需要修改一下）
        current_range = round(current_range + range_step, 3) 
    return None

# 执行动态reset循环
Best_Path = os.path.join('Training', 'Saved Models', 'best_model')
results = dynamic_reset_loop(Best_Path)