import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import ale_py
import ale
import os

# 确保使用与训练时相同的网络结构
class DQN(torch.nn.Module):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 8, stride=4), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4, stride=2), 
            torch.nn.ReLU(),
            torch.nn.Flatten(), 
            torch.nn.Linear(2592, 256), 
            torch.nn.ReLU(),
            torch.nn.Linear(256, nb_actions)
        )

    def forward(self, x):
        return self.network(x / 255.)

def load_checkpoint(model, optimizer=None, path='checkpoint.pth'):
    """加载检查点文件，包含模型参数和优化器状态[2](@ref)"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    reward = checkpoint.get('reward', 0)
    
    print(f"检查点加载成功: 轮次 {epoch}")
    return model, optimizer, epoch

def evaluate_model(env, model, num_episodes=5, device='cuda'):
    """评估模型在环境中的表现[1](@ref)"""
    model.eval() 
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        dead = False
        total_reward = 0
        step_count = 0
        
        obs, info = env.reset()
        
        for _ in range(np.random.randint(1, 30)):
            obs, _, _, _, info = env.step(1)
        
        while not dead:
            current_life = info['lives']
            
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).unsqueeze(0).to(device))
                action = torch.argmax(q_values, dim=1).item()
            
            # 执行动作
            next_obs, reward, dead, _, info = env.step(action)
            
            
            total_reward += reward
            step_count += 1
            obs = next_obs
            
            env.render()
            
            if info['lives'] < current_life:
                    obs, _, _, _, info = env.step(1)
                
            if dead:
                break
        
        episode_rewards.append(total_reward)
        print(f"回合 {episode + 1}: 总奖励 = {total_reward}, 步数 = {step_count}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    
    print(f"\n=== 评估结果 ===")
    print(f"测试回合数: {num_episodes}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"最高奖励: {max_reward}")
    print(f"最低奖励: {min_reward}")
    
    return episode_rewards

def create_environment():
    """创建与训练时相同的环境[1](@ref)"""
    env = gym.make("BreakoutNoFrameskip-v4", render_mode='human')
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = MaxAndSkipEnv(env, skip=4)
    
    return env

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    env = create_environment()
    
    model = DQN(env.action_space.n).to(device)
    
    checkpoint_path = "checkpoint.pth" 
    
    if not os.path.exists(checkpoint_path):
        # 尝试查找最新的检查点文件
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('checkpoint_') and f.endswith('.pth')]
        if checkpoint_files:
            checkpoint_path = max(checkpoint_files, key=os.path.getctime)
            print(f"使用找到的最新检查点: {checkpoint_path}")
        else:
            print("错误: 未找到检查点文件!")
            return
    
    model, _, epoch= load_checkpoint(model, path=checkpoint_path)
    
    print("开始评估模型...")
    rewards = evaluate_model(env, model, num_episodes=5, device=device)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), rewards, 'b-o', linewidth=2, markersize=8)
    plt.title('DQN在Breakout游戏中的表现')
    plt.xlabel('测试回合')
    plt.ylabel('奖励')
    plt.grid(True, alpha=0.3)
    plt.savefig('dqn_breakout_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()