#!/usr/bin/env python3
"""
Simplified Suika AI Demo
Demonstrates AI concepts without requiring TensorFlow.
Shows spatial heuristics, decision scoring, and basic AI logic.
"""

import gymnasium
import suika_env
import numpy as np
import time
from PIL import Image

class SimpleSuikaAI:
    """Simplified AI agent that demonstrates the core concepts without TensorFlow."""
    
    def __init__(self):
        self.exploration_rate = 0.3
        self.spatial_weights = {
            'height_penalty': 0.4,
            'gap_penalty': 0.3,
            'merge_potential': 0.2,
            'edge_preference': 0.1
        }
    
    def _extract_spatial_features(self, image):
        """Extract spatial features from game state image."""
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image
            
        height, width = gray.shape
        
        # 1. Height analysis (penalize high stacks)
        height_map = np.zeros(width)
        for x in range(width):
            for y in range(height):
                if gray[y, x] < 200:  # Non-white pixel (game element)
                    height_map[x] = height - y
                    break
        
        # 2. Gap analysis (penalize gaps between elements)
        gap_map = np.zeros(width)
        for x in range(width):
            gaps = 0
            in_element = False
            for y in range(height):
                if gray[y, x] < 200:  # Game element
                    if not in_element:
                        in_element = True
                    else:
                        gaps += 1
                else:
                    if in_element:
                        in_element = False
            gap_map[x] = gaps
        
        # 3. Merge potential (reward areas where similar colors might merge)
        merge_potential = np.zeros(width)
        for x in range(width):
            if x > 0 and x < width - 1:
                # Check if adjacent columns have similar heights
                left_height = height_map[x-1] if x > 0 else 0
                right_height = height_map[x+1] if x < width-1 else 0
                current_height = height_map[x]
                
                if abs(left_height - current_height) <= 2 or abs(right_height - current_height) <= 2:
                    merge_potential[x] = 1.0
        
        # 4. Edge preference (slight preference for edges)
        edge_preference = np.zeros(width)
        edge_preference[0] = 0.1  # Left edge
        edge_preference[-1] = 0.1  # Right edge
        
        return {
            'height_map': height_map,
            'gap_map': gap_map,
            'merge_potential': merge_potential,
            'edge_preference': edge_preference
        }
    
    def _evaluate_position(self, image, action):
        """Evaluate the quality of a potential action using spatial heuristics."""
        features = self._extract_spatial_features(image)
        
        # Convert action to x-coordinate
        x_pos = int(action[0] * (image.shape[1] - 1))
        
        # Calculate spatial score
        height_penalty = -features['height_map'][x_pos] * self.spatial_weights['height_penalty']
        gap_penalty = -features['gap_map'][x_pos] * self.spatial_weights['gap_penalty']
        merge_bonus = features['merge_potential'][x_pos] * self.spatial_weights['merge_potential']
        edge_bonus = features['edge_preference'][x_pos] * self.spatial_weights['edge_preference']
        
        # Combine scores
        spatial_score = height_penalty + gap_penalty + merge_bonus + edge_bonus
        
        # Add some randomness for exploration
        exploration_bonus = np.random.normal(0, 0.1)
        
        return spatial_score + exploration_bonus
    
    def select_action(self, image):
        """Select action using spatial heuristics and exploration."""
        if np.random.random() < self.exploration_rate:
            # Random exploration
            return np.random.random(1)
        else:
            # Evaluate multiple positions and choose best
            best_score = float('-inf')
            best_action = np.array([0.5])  # Default to center
            
            # Test multiple positions
            for i in range(10):
                test_action = np.random.random(1)
                score = self._evaluate_position(image, test_action)
                
                if score > best_score:
                    best_score = score
                    best_action = test_action
            
            return best_action
    
    def analyze_game_state(self, image):
        """Analyze and display game state information."""
        features = self._extract_spatial_features(image)
        
        print("\n🔍 **Game State Analysis**")
        print("=" * 40)
        print(f"📊 Height Analysis:")
        for i, height in enumerate(features['height_map']):
            if height > 0:
                print(f"   Column {i}: Height {height:.1f}")
        
        print(f"\n🕳️  Gap Analysis:")
        for i, gaps in enumerate(features['gap_map']):
            if gaps > 0:
                print(f"   Column {i}: {gaps:.0f} gaps")
        
        print(f"\n🔗 Merge Potential:")
        for i, potential in enumerate(features['merge_potential']):
            if potential > 0:
                print(f"   Column {i}: High merge potential")
        
        # Calculate overall game state quality
        avg_height = np.mean(features['height_map'])
        total_gaps = np.sum(features['gap_map'])
        merge_opportunities = np.sum(features['merge_potential'])
        
        print(f"\n📈 **Game State Quality Metrics**")
        print(f"   Average Height: {avg_height:.2f}")
        print(f"   Total Gaps: {total_gaps:.0f}")
        print(f"   Merge Opportunities: {merge_opportunities:.0f}")
        
        return features

def demo_basic_ai():
    """Demonstrate the basic AI concepts."""
    
    print("🍉 **Simplified Suika AI Demo**")
    print("=" * 50)
    print("This demo shows the AI concepts without TensorFlow:")
    print("• Spatial evaluation heuristics")
    print("• Decision scoring algorithms")
    print("• Intelligent action selection")
    print()
    
    try:
        # Create environment
        print("🚀 Creating environment...")
        env = gymnasium.make("SuikaEnv-v0")
        print("✅ Environment created!")
        
        # Create AI agent
        print("🤖 Creating AI agent...")
        ai_agent = SimpleSuikaAI()
        print("✅ AI agent created!")
        
        # Run demo
        print("\n🎮 Running AI demo...")
        print("=" * 30)
        
        obs, info = env.reset()
        total_score = 0
        episode_steps = 0
        
        for step in range(10):  # Run for 10 steps
            print(f"\n🔄 **Step {step + 1}**")
            
            # Analyze current game state
            features = ai_agent.analyze_game_state(obs['image'])
            
            # Select action using AI
            action = ai_agent.select_action(obs['image'])
            print(f"🎯 AI Action: {action[0]:.3f} (x-position: {int(action[0] * 640)})")
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Update metrics
            total_score = obs['score'].item()
            episode_steps = step + 1
            
            print(f"📊 Score: {total_score:.0f}, Reward: {reward:.1f}")
            
            if terminated:
                print("🏁 Episode terminated!")
                break
            
            # Small delay to see what's happening
            time.sleep(0.5)
        
        # Final results
        print("\n" + "=" * 50)
        print("🎉 **Demo Complete!**")
        print("=" * 50)
        print(f"📊 Final Score: {total_score:.0f}")
        print(f"🔄 Total Steps: {episode_steps}")
        print(f"🤖 AI Actions Made: {episode_steps}")
        print()
        print("🔍 **What You Just Saw:**")
        print("• Spatial analysis of game state")
        print("• Height and gap evaluation")
        print("• Merge potential assessment")
        print("• Intelligent action selection")
        print("• Decision scoring algorithms")
        print()
        print("🚀 **Next Steps:**")
        print("• Install TensorFlow for full neural network")
        print("• Run training: python3 train_ai_agent.py")
        print("• Compare performance: python3 compare_ai_baseline.py")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("🍉 Suika AI Concepts Demo")
    print("=" * 40)
    print("This demonstrates the AI logic without requiring TensorFlow.")
    print()
    
    success = demo_basic_ai()
    
    if success:
        print("\n🎯 **Demo Summary**")
        print("The AI agent successfully demonstrated:")
        print("✅ Spatial evaluation heuristics")
        print("✅ Decision scoring algorithms")
        print("✅ Intelligent action selection")
        print("✅ Game state analysis")
        print()
        print("This shows the core AI concepts from your resume!")
    else:
        print("\n❌ Demo encountered issues.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 