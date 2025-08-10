#!/usr/bin/env python3
"""
Suika Game AI Agent Training Script
Demonstrates the deep learning agent training process with performance tracking.
"""

import gymnasium
import suika_env
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
import json
import os

def train_ai_agent(episodes: int = 100, 
                   training_interval: int = 10,
                   save_model: bool = True,
                   model_path: str = "suika_ai_model.h5"):
    """
    Train the Suika AI agent and track performance improvements.
    
    Args:
        episodes: Number of training episodes
        training_interval: How often to retrain the model
        save_model: Whether to save the trained model
        model_path: Path to save/load the model
    """
    
    print("🚀 Starting Suika Game AI Agent Training")
    print("=" * 50)
    
    # Create base environment
    base_env = gymnasium.make("SuikaEnv-v0")
    
    # Create AI agent
    ai_agent = suika_env.SuikaAIAgent(
        base_env=base_env,
        model_path=model_path if os.path.exists(model_path) else None,
        learning_rate=0.001,
        exploration_rate=0.2,  # Start with higher exploration
        simulation_depth=3
    )
    
    # Training data collection
    training_data: List[Tuple] = []
    episode_rewards = []
    episode_scores = []
    performance_history = []
    
    print(f"📊 Baseline Score: {ai_agent.baseline_score}")
    print(f"🎯 Target Score: {ai_agent.target_score}")
    print(f"🎮 Training for {episodes} episodes")
    print()
    
    start_time = time.time()
    
    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}", end=" ")
        
        # Reset environment
        obs, _ = ai_agent.reset()
        episode_reward = 0
        step_count = 0
        
        # Play episode
        terminated = False
        while not terminated:
            # AI agent selects action
            action = ai_agent._select_action(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = ai_agent.step(action)
            
            # Store training data
            training_data.append((obs['image'], action[0], reward))
            episode_reward += reward
            step_count += 1
            
            # Early termination if game is too long
            if step_count > 200:
                break
        
        # Episode completed
        final_score = obs['score'].item()
        episode_scores.append(final_score)
        episode_rewards.append(episode_reward)
        
        # Calculate running average
        if len(episode_scores) >= 10:
            recent_avg = np.mean(episode_scores[-10:])
            improvement = ((recent_avg - ai_agent.baseline_score) / ai_agent.baseline_score) * 100
            print(f"Score: {final_score:.0f} | Avg: {recent_avg:.0f} | Improvement: {improvement:+.1f}%")
        else:
            print(f"Score: {final_score:.0f}")
        
        # Retrain model periodically
        if (episode + 1) % training_interval == 0 and len(training_data) > 0:
            print(f"  🔄 Retraining model with {len(training_data)} samples...")
            
            # Use recent training data
            recent_data = training_data[-1000:] if len(training_data) > 1000 else training_data
            
            # Train the model
            ai_agent.train(recent_data, epochs=5)
            
            # Clear old training data to prevent memory issues
            training_data = recent_data[-500:]
        
        # Track performance
        if len(episode_scores) >= 10:
            performance_stats = ai_agent.get_performance_stats()
            performance_history.append(performance_stats)
            
            # Check if target achieved
            if performance_stats['target_achieved']:
                print(f"  🎉 Target achieved! Average score: {performance_stats['average_score']:.0f}")
                break
    
    training_time = time.time() - start_time
    
    # Final performance analysis
    print("\n" + "=" * 50)
    print("🏆 TRAINING COMPLETED")
    print("=" * 50)
    
    final_stats = ai_agent.get_performance_stats()
    
    if final_stats:
        print(f"📈 Final Average Score: {final_stats['average_score']:.0f}")
        print(f"📊 Baseline Score: {final_stats['baseline_score']:.0f}")
        print(f"🚀 Improvement: {final_stats['improvement_percentage']:+.1f}%")
        print(f"🎯 Target Achieved: {'✅ YES' if final_stats['target_achieved'] else '❌ NO'}")
        print(f"🎮 Episodes Played: {final_stats['episodes_played']}")
    
    print(f"⏱️  Total Training Time: {training_time:.1f} seconds")
    print(f"📊 Episodes per Minute: {episodes / (training_time / 60):.1f}")
    
    # Save model if requested
    if save_model and final_stats and final_stats['target_achieved']:
        ai_agent.save_model(model_path)
        print(f"💾 Model saved to: {model_path}")
    
    # Plot performance
    plot_performance(episode_scores, episode_rewards, performance_history)
    
    # Save performance data
    save_performance_data(episode_scores, episode_rewards, performance_history)
    
    # Cleanup
    ai_agent.close()
    
    return final_stats

def plot_performance(episode_scores: List[float], 
                    episode_rewards: List[float], 
                    performance_history: List[Dict]):
    """Plot training performance metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Suika Game AI Agent Training Performance', fontsize=16, fontweight='bold')
    
    # Episode scores
    axes[0, 0].plot(episode_scores, alpha=0.7, color='blue')
    axes[0, 0].set_title('Episode Scores')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Running average scores
    if len(episode_scores) >= 10:
        running_avg = [np.mean(episode_scores[max(0, i-9):i+1]) for i in range(len(episode_scores))]
        axes[0, 0].plot(running_avg, color='red', linewidth=2, label='10-Episode Average')
        axes[0, 0].legend()
    
    # Episode rewards
    axes[0, 1].plot(episode_rewards, alpha=0.7, color='green')
    axes[0, 1].set_title('Episode Rewards')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance improvement over baseline
    if performance_history:
        episodes = list(range(len(performance_history)))
        improvements = [stats['improvement_percentage'] for stats in performance_history]
        axes[1, 0].plot(episodes, improvements, color='orange', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Improvement Over Baseline')
        axes[1, 0].set_xlabel('Training Interval')
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Target achievement line
        target_achieved = [stats['target_achieved'] for stats in performance_history]
        target_episodes = [i for i, achieved in enumerate(target_achieved) if achieved]
        if target_episodes:
            axes[1, 0].axvline(x=target_episodes[0], color='red', linestyle=':', 
                              label=f'Target Achieved (Episode {target_episodes[0] * 10})')
            axes[1, 0].legend()
    
    # Score distribution
    if episode_scores:
        axes[1, 1].hist(episode_scores, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(x=np.mean(episode_scores), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(episode_scores):.0f}')
        axes[1, 1].axvline(x=3000, color='blue', linestyle='--', 
                           label='Baseline: 3000')
        axes[1, 1].axvline(x=3500, color='green', linestyle='--', 
                           label='Target: 3500')
        axes[1, 1].set_title('Score Distribution')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('suika_ai_training_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_performance_data(episode_scores: List[float], 
                         episode_rewards: List[float], 
                         performance_history: List[Dict]):
    """Save performance data to JSON file."""
    
    data = {
        'episode_scores': episode_scores,
        'episode_rewards': episode_rewards,
        'performance_history': performance_history,
        'summary': {
            'total_episodes': len(episode_scores),
            'final_average_score': np.mean(episode_scores[-10:]) if len(episode_scores) >= 10 else np.mean(episode_scores),
            'baseline_score': 3000,
            'target_score': 3500,
            'improvement_percentage': ((np.mean(episode_scores[-10:]) - 3000) / 3000 * 100) if len(episode_scores) >= 10 else 0
        }
    }
    
    with open('suika_ai_performance.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Performance data saved to: suika_ai_performance.json")

def demo_ai_agent(episodes: int = 5, model_path: str = "suika_ai_model.h5"):
    """Demonstrate the trained AI agent playing the game."""
    
    print("🎮 Demonstrating Trained AI Agent")
    print("=" * 40)
    
    # Create base environment
    base_env = gymnasium.make("SuikaEnv-v0")
    
    # Create AI agent
    ai_agent = suika_env.SuikaAIAgent(
        base_env=base_env,
        model_path=model_path if os.path.exists(model_path) else None,
        exploration_rate=0.0  # No exploration during demo
    )
    
    total_score = 0
    
    for episode in range(episodes):
        print(f"Demo Episode {episode + 1}/{episodes}")
        
        obs, _ = ai_agent.reset()
        episode_score = 0
        step_count = 0
        
        terminated = False
        while not terminated:
            # AI agent selects action
            action = ai_agent._select_action(obs)
            
            # Take step
            obs, reward, terminated, truncated, info = ai_agent.step(action)
            
            episode_score += reward
            step_count += 1
            
            if step_count > 200:
                break
        
        final_score = obs['score'].item()
        total_score += final_score
        
        print(f"  Final Score: {final_score:.0f}")
        print(f"  Steps: {step_count}")
        print()
    
    avg_score = total_score / episodes
    improvement = ((avg_score - 3000) / 3000) * 100
    
    print(f"📊 Demo Results:")
    print(f"  Average Score: {avg_score:.0f}")
    print(f"  Improvement over Baseline: {improvement:+.1f}%")
    print(f"  Target Achieved: {'✅ YES' if avg_score >= 3500 else '❌ NO'}")
    
    ai_agent.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Suika Game AI Agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--training-interval', type=int, default=10, help='Training interval')
    parser.add_argument('--demo', action='store_true', help='Run demo after training')
    parser.add_argument('--model-path', type=str, default='suika_ai_model.h5', help='Model save/load path')
    
    args = parser.parse_args()
    
    try:
        # Train the agent
        final_stats = train_ai_agent(
            episodes=args.episodes,
            training_interval=args.training_interval,
            model_path=args.model_path
        )
        
        # Run demo if requested
        if args.demo and final_stats and final_stats.get('target_achieved', False):
            print("\n" + "=" * 50)
            demo_ai_agent(episodes=5, model_path=args.model_path)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc() 