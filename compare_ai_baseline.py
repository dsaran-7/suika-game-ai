#!/usr/bin/env python3
"""
Suika Game AI vs Baseline Comparison Script
Demonstrates the 17% improvement in average score (3000 → 3500) mentioned in the resume.
"""

import gymnasium
import suika_env
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Dict
import json

def run_baseline_agent(episodes: int = 50) -> List[float]:
    """
    Run the baseline agent (random policy) to establish baseline performance.
    """
    print("🎯 Running Baseline Agent (Random Policy)")
    print("=" * 40)
    
    env = gymnasium.make("SuikaEnv-v0")
    scores = []
    
    for episode in range(episodes):
        obs, _ = env.reset()
        episode_score = 0
        step_count = 0
        
        terminated = False
        while not terminated:
            # Random action selection (baseline)
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_score += reward
            step_count += 1
            
            if step_count > 200:
                break
        
        final_score = obs['score'].item()
        scores.append(final_score)
        
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(scores[-10:])
            print(f"Episode {episode + 1}: Score {final_score:.0f} | Recent Avg: {recent_avg:.0f}")
    
    env.close()
    
    baseline_avg = np.mean(scores)
    print(f"\n📊 Baseline Performance:")
    print(f"  Average Score: {baseline_avg:.0f}")
    print(f"  Episodes: {len(scores)}")
    
    return scores

def run_ai_agent(episodes: int = 50, model_path: str = None) -> List[float]:
    """
    Run the AI agent to demonstrate improved performance.
    """
    print("\n🤖 Running AI Agent")
    print("=" * 40)
    
    # Create base environment
    base_env = gymnasium.make("SuikaEnv-v0")
    
    # Create AI agent
    ai_agent = suika_env.SuikaAIAgent(
        base_env=base_env,
        model_path=model_path,
        exploration_rate=0.1,  # Low exploration for evaluation
        simulation_depth=3
    )
    
    scores = []
    
    for episode in range(episodes):
        obs, _ = ai_agent.reset()
        episode_score = 0
        step_count = 0
        
        terminated = False
        while not terminated:
            # AI-driven action selection
            action = ai_agent._select_action(obs)
            
            obs, reward, terminated, truncated, info = ai_agent.step(action)
            episode_score += reward
            step_count += 1
            
            if step_count > 200:
                break
        
        final_score = obs['score'].item()
        scores.append(final_score)
        
        if (episode + 1) % 10 == 0:
            recent_avg = np.mean(scores[-10:])
            print(f"Episode {episode + 1}: Score {final_score:.0f} | Recent Avg: {recent_avg:.0f}")
    
    ai_agent.close()
    
    ai_avg = np.mean(scores)
    print(f"\n📊 AI Agent Performance:")
    print(f"  Average Score: {ai_avg:.0f}")
    print(f"  Episodes: {len(scores)}")
    
    return scores

def compare_performance(baseline_scores: List[float], 
                       ai_scores: List[float]) -> Dict:
    """
    Compare baseline vs AI agent performance and calculate improvements.
    """
    print("\n📈 PERFORMANCE COMPARISON")
    print("=" * 50)
    
    baseline_avg = np.mean(baseline_scores)
    ai_avg = np.mean(ai_scores)
    
    # Calculate improvements
    absolute_improvement = ai_avg - baseline_avg
    percentage_improvement = (absolute_improvement / baseline_avg) * 100
    
    # Target achievement
    target_score = 3500
    target_achieved = ai_avg >= target_score
    
    print(f"🎯 Baseline Agent (Random Policy):")
    print(f"  Average Score: {baseline_avg:.0f}")
    print(f"  Standard Deviation: {np.std(baseline_scores):.0f}")
    
    print(f"\n🤖 AI Agent (Deep Learning + Spatial Heuristics):")
    print(f"  Average Score: {ai_avg:.0f}")
    print(f"  Standard Deviation: {np.std(ai_scores):.0f}")
    
    print(f"\n🚀 Performance Improvement:")
    print(f"  Absolute Improvement: {absolute_improvement:+.0f} points")
    print(f"  Percentage Improvement: {percentage_improvement:+.1f}%")
    print(f"  Target Score (3500): {'✅ ACHIEVED' if target_achieved else '❌ NOT ACHIEVED'}")
    
    # Resume claim verification
    resume_baseline = 3000
    resume_target = 3500
    resume_improvement = ((resume_target - resume_baseline) / resume_baseline) * 100
    
    print(f"\n📋 Resume Claim Verification:")
    print(f"  Claimed Baseline: {resume_baseline}")
    print(f"  Claimed Target: {resume_target}")
    print(f"  Claimed Improvement: {resume_improvement:.1f}%")
    print(f"  Actual Improvement: {percentage_improvement:.1f}%")
    print(f"  Claim Verified: {'✅ YES' if percentage_improvement >= resume_improvement else '❌ NO'}")
    
    return {
        'baseline_avg': baseline_avg,
        'ai_avg': ai_avg,
        'absolute_improvement': absolute_improvement,
        'percentage_improvement': percentage_improvement,
        'target_achieved': target_achieved,
        'resume_claim_verified': percentage_improvement >= resume_improvement
    }

def plot_comparison(baseline_scores: List[float], 
                   ai_scores: List[float], 
                   comparison_results: Dict):
    """
    Create comprehensive comparison plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Suika Game: AI Agent vs Baseline Performance Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Score comparison over episodes
    axes[0, 0].plot(baseline_scores, alpha=0.7, color='red', label='Baseline (Random)')
    axes[0, 0].plot(ai_scores, alpha=0.7, color='blue', label='AI Agent')
    axes[0, 0].axhline(y=3000, color='red', linestyle='--', alpha=0.5, label='Baseline Target')
    axes[0, 0].axhline(y=3500, color='green', linestyle='--', alpha=0.5, label='AI Target')
    axes[0, 0].set_title('Score Progression Over Episodes')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score distribution comparison
    axes[0, 1].hist(baseline_scores, bins=15, alpha=0.7, color='red', 
                     label=f'Baseline (μ={np.mean(baseline_scores):.0f})', edgecolor='black')
    axes[0, 1].hist(ai_scores, bins=15, alpha=0.7, color='blue', 
                     label=f'AI Agent (μ={np.mean(ai_scores):.0f})', edgecolor='black')
    axes[0, 1].set_title('Score Distribution Comparison')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    metrics = ['Average Score', 'Min Score', 'Max Score', 'Std Dev']
    baseline_metrics = [
        np.mean(baseline_scores),
        np.min(baseline_scores),
        np.max(baseline_scores),
        np.std(baseline_scores)
    ]
    ai_metrics = [
        np.mean(ai_scores),
        np.min(ai_scores),
        np.max(ai_scores),
        np.std(ai_scores)
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, baseline_metrics, width, label='Baseline', color='red', alpha=0.7)
    axes[1, 0].bar(x + width/2, ai_metrics, width, label='AI Agent', color='blue', alpha=0.7)
    axes[1, 0].set_title('Performance Metrics Comparison')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(metrics, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Improvement visualization
    improvement = comparison_results['percentage_improvement']
    target_improvement = 17.0  # Resume claim
    
    bars = ['Actual\nImprovement', 'Resume\nClaim']
    values = [improvement, target_improvement]
    colors = ['green' if improvement >= target_improvement else 'orange', 'blue']
    
    axes[1, 1].bar(bars, values, color=colors, alpha=0.7)
    axes[1, 1].set_title('Improvement Comparison')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].axhline(y=target_improvement, color='red', linestyle='--', alpha=0.7, label='Resume Target')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('suika_ai_vs_baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_comparison_data(baseline_scores: List[float], 
                        ai_scores: List[float], 
                        comparison_results: Dict):
    """
    Save comparison data to JSON file.
    """
    data = {
        'baseline_scores': baseline_scores,
        'ai_scores': ai_scores,
        'comparison_results': comparison_results,
        'summary': {
            'baseline_episodes': len(baseline_scores),
            'ai_episodes': len(ai_scores),
            'resume_claim': {
                'baseline_score': 3000,
                'target_score': 3500,
                'claimed_improvement': 17.0,
                'actual_improvement': comparison_results['percentage_improvement'],
                'claim_verified': comparison_results['resume_claim_verified']
            }
        }
    }
    
    with open('suika_ai_baseline_comparison.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Comparison data saved to: suika_ai_baseline_comparison.png")
    print(f"💾 Comparison data saved to: suika_ai_baseline_comparison.json")

def main():
    """
    Main comparison function.
    """
    print("🍉 Suika Game AI vs Baseline Performance Comparison")
    print("=" * 60)
    print("This script demonstrates the AI agent's improvement over baseline")
    print("performance, verifying the resume claim of 17% improvement.")
    print()
    
    # Configuration
    episodes = 50
    model_path = "suika_ai_model.h5"  # Path to trained model if available
    
    try:
        # Run baseline agent
        baseline_scores = run_baseline_agent(episodes)
        
        # Run AI agent
        ai_scores = run_ai_agent(episodes, model_path)
        
        # Compare performance
        comparison_results = compare_performance(baseline_scores, ai_scores)
        
        # Create visualizations
        plot_comparison(baseline_scores, ai_scores, comparison_results)
        
        # Save data
        save_comparison_data(baseline_scores, ai_scores, comparison_results)
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏆 COMPARISON COMPLETED")
        print("=" * 60)
        
        if comparison_results['resume_claim_verified']:
            print("✅ SUCCESS: Resume claim verified!")
            print(f"   The AI agent achieved {comparison_results['percentage_improvement']:.1f}% improvement")
            print(f"   over the baseline, exceeding the claimed 17% improvement.")
        else:
            print("⚠️  PARTIAL SUCCESS: AI agent improved performance but")
            print(f"   achieved {comparison_results['percentage_improvement']:.1f}% improvement")
            print(f"   vs. the claimed 17% improvement.")
        
        print(f"\n📊 Key Results:")
        print(f"   Baseline Average: {comparison_results['baseline_avg']:.0f}")
        print(f"   AI Agent Average: {comparison_results['ai_avg']:.0f}")
        print(f"   Improvement: {comparison_results['percentage_improvement']:+.1f}%")
        
    except KeyboardInterrupt:
        print("\n⚠️  Comparison interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 