import numpy as np
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import time
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from gymnasium.envs.classic_control import utils
from typing import Optional
import importlib
import sys
import matplotlib.pyplot as plt
from collections import deque
import time

##Set the basic parameters
balance_time = 20  # 平衡任务的总时长(秒)
h_in = 1/100       # 时间步长(0.01秒，即每秒100步)

# 定义平衡判断函数（目前的设定是0.1范围之内，则符合要求）
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

def create_dynamic_reset_method(pos_range, theta_range, vel_range, reg_range):
    """
    创建新的reset方法，分别设置位置和速度的范围
    
    Args:
        pos_range: 位置变量 x 的范围
        theta_range: 角度变量 theta 的范围
        vel_range: 线速度变量 x_dot 的范围
        reg_range: 角速度变量 theta_dot 的范围
    """
    print(f"🔧 创建动态reset方法:")
    print(f"   位置范围 x : [{-pos_range}, {pos_range}]")
    print(f"   角度范围 θ : [{-theta_range}, {theta_range}]")
    print(f"   线速度范围 x_dot : [{-vel_range}, {vel_range}]")
    print(f"   角速度范围 θ_dot : [{-reg_range}, {reg_range}]")
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # 调用父类的reset（但不传seed，避免重复设置）
        super(type(self), self).reset(seed=seed)
        
        # 生成位置变量 (x, theta)
        x = self.np_random.uniform(low=-pos_range, high=pos_range)
        theta = self.np_random.uniform(low=-theta_range, high=theta_range)
        
        # 生成速度变量 (x_dot, theta_dot)  
        x_dot = self.np_random.uniform(low=-vel_range, high=vel_range)
        theta_dot = self.np_random.uniform(low=-reg_range, high=reg_range)
        
        # 组合状态向量: [x, x_dot, theta, theta_dot]
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
            
        # 🔍 调试输出：验证生成的状态
        print(f" 生成状态: x={x:.3f}, θ={theta:.3f}, x_dot={x_dot:.3f}, θ_dot={theta_dot:.3f}")
        
        return np.array(self.state, dtype=np.float32), {}
    
    return reset

def update_environment_reset_range(pos_range, theta_range, vel_range, reg_range):
    """
    更新环境的reset范围，分别设置位置和速度范围
    
    Args:
        pos_range: 位置变量 x 的范围
        theta_range: 角度变量 theta 的范围  
        vel_range: 线速度变量 x_dot 的范围
        reg_range: 角速度变量 theta_dot 的范围
    """
    import myCartpoleF
    import inspect
    # 检查原始reset方法
    original_reset = myCartpoleF.myCartPoleEnvF.reset
    
    # 创建新的reset方法
    new_reset_method = create_dynamic_reset_method(pos_range, theta_range, vel_range, reg_range)
    
    # 替换类方法
    myCartpoleF.myCartPoleEnvF.reset = new_reset_method
    
    # 验证替换是否成功
    updated_reset = myCartpoleF.myCartPoleEnvF.reset
    return None

# 模型测试函数
def run_model_test(Test_Path, test_env, model_name="unknown"):
    print(f"\n 开始100000次测试: {model_name} 模型")
    print("="*60)
    
    # 使用传入的预配置环境
    print("🎮 使用预配置的测试环境...")
    env = test_env
    model = PPO.load(Test_Path, env=env)
    
    # 📊 初始化统计
    episode_results = []  # 只记录成功失败标记
    total_successful = 0
    
    print("🎮 开始100000次episode测试...")
    
    # 10000次测试循环
    for i in range(100000):
        obs = env.reset()
        
        # 🔧 解析初始观测值
        if obs.ndim == 2:
            initial_state = obs[0]
        else:
            initial_state = obs
        
        x, x_dot, theta, theta_dot = initial_state
        if i % 1000 == 0:  # 减少输出频率
            print(f"Episode {i+1:5d}: 初始状态 x={x:.3f}, θ={theta:.3f}, x_dot={x_dot:.3f}, θ_dot={theta_dot:.3f}")

        start_time = time.time()
        step_count = 0
        done = False
        balanced_steps = 0  # 记录平衡步数
        
        # 运行单个episode
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
            
            # 评估当前平衡状态
            balance_quality, balance_status, is_balanced = check_balance_status_fixed(x, theta, x_dot, theta_dot)
            
            # 统计平衡步数
            if balance_quality >= 40:  # 简化为只统计>=40质量的步数
                balanced_steps += 1
        
        # 计算episode结果
        episode_balance_rate = (balanced_steps / step_count) * 100 if step_count > 0 else 0
        
        # 判断episode成功与否
        if episode_balance_rate >= 40 and step_count >= 1000:
            success_label = 1
            total_successful += 1
        else:
            success_label = 0
            
        # 只保存成功标记
        episode_results.append(success_label)
        
        # 🖥️ 实时输出进度（每100次输出一次）
        if i % 100 == 0 or i < 10:
            status_icon = "✅" if success_label else "❌"
            print(f"  {status_icon} Episode {i+1:5d}: 步数:{step_count:4d} | 平衡率:{episode_balance_rate:5.1f}%")

    # 📊 统计评估：计算成功率的均值和置信区间
    print(f"\n{'='*40}")
    print(f"📊 {model_name} 模型统计评估结果")
    print(f"{'='*40}")
    
    # 提取成功标记并计算统计量
    labels = np.array(episode_results)
    phat = labels.mean()
    success_count = labels.sum()
    total_count = len(labels)
    
    # 计算95%置信区间 (Wilson方法)
    ci_low, ci_up = proportion_confint(success_count, total_count, alpha=0.05, method='wilson')
    
    # 输出结果
    print(f" 总测试次数: {total_count}")
    print(f" 成功次数: {success_count}")
    print(f" 成功率: {phat:.3%}")
    print(f" 95% 置信区间: {ci_low:.3%} – {ci_up:.3%}")

    env.close()
    
    # 🔄 返回结果给主循环使用（包含统计数据）
    return {
        'model_name': model_name,
        'success_rate': phat,
        'success_count': success_count,
        'total_count': total_count,
        'ci_low': ci_low,
        'ci_up': ci_up,
        'labels': labels  # 返回原始标记数组用于比较
    }

# 模型比较和假设检验函数
def compare_models_and_decide(in_result, be_result, alpha=0.05):
    """
    比较两个模型的性能并进行假设检验
    
    Args:
        in_result: in_model的测试结果
        be_result: be_model的测试结果 
        alpha: 显著性水平，默认0.05
    
    Returns:
        返回字典包含:
        - if_still: True表示需要继续训练，False表示达到最佳效果
        - old_model_stats: Old模型的统计量
        - new_model_stats: New模型的统计量
        - hypothesis_test: 假设检验结果
    """
    print(f"\n{'='*60}")
    print("模型性能比较与假设检验")
    print(f"{'='*60}")
    time.sleep(1)  # 确保输出清晰
    
    from statsmodels.stats.proportion import proportions_ztest
    
    # 提取数据
    in_mean = in_result['success_rate']
    be_mean = be_result['success_rate'] 
    in_count = in_result['success_count']
    be_count = be_result['success_count']
    in_total = in_result['total_count']
    be_total = be_result['total_count']
    
    # 📊 计算标准差 (基于二项分布)
    # 标准差公式: sqrt(p * (1-p) / n)
    old_std = np.sqrt(in_mean * (1 - in_mean) / in_total)
    new_std = np.sqrt(be_mean * (1 - be_mean) / be_total)
    
    # 输出基本信息
    print(f" Old模型成功率: {in_mean:.3%} ({in_count}/{in_total})")
    print(f" Old模型标准差: {old_std:.4f}")
    print(f" New模型成功率: {be_mean:.3%} ({be_count}/{be_total})")
    print(f" New模型标准差: {new_std:.4f}")
    print(f" 差异: {be_mean - in_mean:.3%} (New - Old)")
    time.sleep(1)  # 确保输出清晰
    
    # 计算95%置信区间
    from statsmodels.stats.proportion import proportion_confint
    in_ci = proportion_confint(in_count, in_total, alpha=0.05, method='wilson')
    be_ci = proportion_confint(be_count, be_total, alpha=0.05, method='wilson')
    
    print(f" Old模型95%置信区间: [{in_ci[0]:.3%}, {in_ci[1]:.3%}]")
    print(f" New模型95%置信区间: [{be_ci[0]:.3%}, {be_ci[1]:.3%}]")
    time.sleep(1)  # 确保输出清晰
    
    # 进行双比例假设检验
    # H0: be_mean <= in_mean (BE模型不优于IN模型)  
    # H1: be_mean > in_mean (BE模型优于IN模型)
    try:
        z_stat, p_value = proportions_ztest(
            [be_count, in_count],
            [be_total, in_total], 
            alternative='larger'  # 单边检验：BE > IN
        )
        
        print(f"  假设检验结果:")
        print(f"  H0: New模型成功率 ≤ Old模型成功率")
        print(f"  H1: New模型成功率 > Old模型成功率")
        print(f"  Z统计量: {z_stat:.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  显著性水平: {alpha}")
        time.sleep(1)  # 确保输出清晰
        
        # 添加统计学解释
        if z_stat > 0:
            print(f" Z > 0: New模型成功率更高")
        else:
            print(f" Z ≤ 0: New模型成功率不如Old模型")
            
        print(f" 解释: P值表示在H0为真的条件下，观察到当前差异或更大差异的概率")
        time.sleep(1)  # 确保输出清晰
        
        # 判断结果
        if p_value < alpha:
            print(f"\n 结论: New模型显著优于Old模型 (p < {alpha})")
            print(" 建议: 需要继续训练")
            if_still = True
        else:
            print(f"\n 结论: New模型未显著优于Old模型 (p ≥ {alpha})")  
            print(" 建议: 达到最佳效果，请将当前模型保存，并扩张initial_condition进行处理")
            if_still = False
        
    except Exception as e:
        print(f" 假设检验计算错误: {e}")
        print(" 默认建议继续训练")
        if_still = True
    
    print(f"{'='*60}")
    
    # 🎯 构建返回的统计量字典
    result_stats = {
        'if_still': if_still,
        'old_model_stats': {
            'mean': in_mean,                    # Old模型均值(成功率)
            'std': old_std,                     # Old模型标准差
            'ci_lower': in_ci[0],               # Old模型置信区间下界
            'ci_upper': in_ci[1],               # Old模型置信区间上界
        }
    }
    
    return result_stats

# 主循环：比较两个模型的性能
def model_comparison_loop():
    """
    比较in_model和be_model的性能
    """
    print("开始模型性能比较测试")
    
    # 配置测试环境
    print(f" 配置测试环境:")
    print(f"   位置范围: ±{0.2}")
    print(f"   速度范围: ±{10}")
    update_environment_reset_range(0.2, 0.2, 10,10)
    
    # 注册测试环境
    try:
        env_name = 'CartPoleTest'
        gym.register(
            id=env_name,
            entry_point='myCartpoleF:myCartPoleEnvF',
            reward_threshold=balance_time / h_in * 0.95,
            max_episode_steps=int(balance_time / h_in)
        )
        print(f"✅ 测试环境注册成功: {env_name}")
    except Exception as e:
        print(f"⚠️ 测试环境可能已注册: {e}")
    
    # 创建测试环境实例
    print("创建测试环境实例...")
    test_env = gym.make(env_name, render_mode=None)
    test_env = DummyVecEnv([lambda: test_env])
    
    # 模型路径设置
    in_model_path = os.path.join('Training', 'Saved Models', 'old_model')
    be_model_path = os.path.join('Training', 'Saved Models', 'new_model')
    
    # 检查模型文件是否存在
    if not os.path.exists(in_model_path + '.zip'):
        print(f" Old Model文件不存在: {in_model_path}.zip")
        return True  # 默认继续训练
        
    if not os.path.exists(be_model_path + '.zip'):
        print(f" New Model文件不存在: {be_model_path}.zip")
        return True  # 默认继续训练
    
    # 测试Old模型（使用预配置的测试环境）
    print(f"\n 测试Old模型...")
    in_result = run_model_test(in_model_path, test_env, "Old")
    
    # 测试New模型（使用预配置的测试环境）
    print(f"\n 测试New模型...")
    be_result = run_model_test(be_model_path, test_env, "New")
    
    # 比较模型并做决策
    comparison_result = compare_models_and_decide(in_result, be_result)
    
    # 提取决策结果
    if_still = comparison_result['if_still']
    
    # 构建最终统计结果
    return if_still, comparison_result

# 执行模型比较（主要功能）
# stats_result里面包含了old_model的平均值，标准差，置信区间上下界四个数据
print(" 执行模型性能比较...")
if_still_training, stats_result = model_comparison_loop()
print(f"\n 模型比较结果: {stats_result}")
if if_still_training:
    print(f"\n 最终结论: 需要继续训练")
else:
    print(f"\n 最终结论: 训练已达到最佳效果")