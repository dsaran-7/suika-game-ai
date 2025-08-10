#!/usr/bin/env python3
"""
Suika Game AI - Quick Start Script
Demonstrates the key features of the AI agent in a simple format.
"""

import gymnasium
import suika_env
import numpy as np
import time
import argparse
import os

def quick_demo(model_path=None):
    """Quick demonstration of the AI agent capabilities."""
    
    print("🍉 Suika Game AI - Quick Start Demo")
    print("=" * 40)
    
    if model_path:
        print(f"🤖 Loading trained model from: {model_path}")
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("   Running with untrained model instead...")
            model_path = None
    else:
        print("🤖 Running with untrained model (random weights)")
        print("   This demonstrates AI concepts but won't achieve target scores")
        print("   Train the model first with: python3 train_ai_agent.py")
    
    try:
        # Create base environment with visible browser
        print("🚀 Creating Suika environment...")
        base_env = gymnasium.make("SuikaEnv-v0", headless=False, delay_before_img_capture=1.0)
        
        # Create AI agent
        print("🤖 Initializing AI agent...")
        ai_agent = suika_env.SuikaAIAgent(
            base_env=base_env,
            exploration_rate=0.3,  # Higher exploration for demo
            simulation_depth=2
        )
        
        # Load trained model if specified
        if model_path:
            try:
                ai_agent.load_model(model_path)
                print("✅ Trained model loaded successfully!")
                print("   Expected performance: 3500+ score (~17% improvement)")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                print("   Running with untrained model instead...")
        else:
            print("✅ Using untrained model (demonstrates AI concepts)")
            print("   Expected performance: ~3000 score (baseline)")
        
        print("✅ AI agent ready!")
        print(f"📊 Baseline target: {ai_agent.baseline_score}")
        print(f"🎯 Improvement target: {ai_agent.target_score}")
        print()
        
        # Run a few episodes to demonstrate
        episodes = 3
        total_score = 0
        
        for episode in range(episodes):
            print(f"🎮 Episode {episode + 1}/{episodes}")
            
            obs, _ = ai_agent.reset()
            episode_score = 0
            step_count = 0
            
            terminated = False
            while not terminated and step_count < 300:  # Allow longer gameplay to see fruits fall
                # AI agent selects action
                action = ai_agent._select_action(obs)
                
                # Take step
                obs, reward, terminated, truncated, info = ai_agent.step(action)
                episode_score += reward
                step_count += 1
                
                # Show progress
                if step_count % 10 == 0:  # More frequent updates
                    current_score = obs['score'].item()
                    print(f"  Step {step_count}: Score {current_score:.0f}")
                    
                    # Show AI thinking process
                    if 'image' in obs:
                        spatial_features = ai_agent._extract_spatial_features(obs['image'])
                        if 'fruit_sizes' in spatial_features:
                            fruit_info = spatial_features['fruit_sizes']
                            print(f"    🍎 Stacking Quality: {fruit_info['total_stacking_quality']:.2f}")
                            print(f"    🎯 Small Fruit Opportunities: {fruit_info['small_fruit_opportunities']}")
                
                # Add small delay to see the game in action
                import time
                time.sleep(0.5)
            
            final_score = obs['score'].item()
            total_score += final_score
            
            print(f"  Final Score: {final_score:.0f}")
            print(f"  Steps: {step_count}")
            print()
        
        # Show results
        avg_score = total_score / episodes
        improvement = ((avg_score - ai_agent.baseline_score) / ai_agent.baseline_score) * 100
        
        print("📊 Demo Results:")
        print(f"  Episodes played: {episodes}")
        print(f"  Average score: {avg_score:.0f}")
        print(f"  Baseline score: {ai_agent.baseline_score}")
        print(f"  Improvement: {improvement:+.1f}%")
        print(f"  Target achieved: {'✅ YES' if avg_score >= ai_agent.target_score else '❌ NO'}")
        
        # Show spatial features from last observation
        if 'image' in obs:
            print(f"\n🔍 Spatial Analysis (Last Game State):")
            spatial_features = ai_agent._extract_spatial_features(obs['image'])
            for feature, value in spatial_features.items():
                print(f"  {feature}: {value:.2f}")
        
        # Cleanup
        ai_agent.close()
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💡 To run full training: python train_ai_agent.py")
        print(f"📊 To compare vs baseline: python compare_ai_baseline.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

def show_ai_capabilities():
    """Display the AI agent's key capabilities."""
    
    print("\n🧠 AI Agent Capabilities")
    print("=" * 30)
    
    capabilities = [
        "✅ TensorFlow Sequential API with custom loss functions",
        "✅ Convolutional Neural Network for spatial reasoning",
        "✅ Spatial evaluation heuristics (height, gaps, edges)",
        "✅ Minimax-inspired decision scoring",
        "✅ Automated simulation loops for forward planning",
        "✅ Performance tracking and improvement metrics",
        "✅ Configurable exploration and training parameters",
        "✅ Model persistence and transfer learning"
    ]
    
    for capability in capabilities:
        print(capability)
    
    print(f"\n🎯 Performance Targets:")
    print(f"  Baseline Score: 3000 (random policy)")
    print(f"  Target Score: 3500 (AI agent)")
    print(f"  Expected Improvement: ~17%")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Suika Game AI - Quick Start Demo")
    parser.add_argument("--model-path", type=str, help="Path to trained model file (.h5)")
    parser.add_argument("--auto-run", action="store_true", help="Run demo automatically without prompting")
    args = parser.parse_args()
    
    print("Welcome to Suika Game AI!")
    print("This demo shows the key features of the deep learning agent.")
    print()
    
    show_ai_capabilities()
    print()
    
    # Check if model path was provided
    if args.model_path:
        print(f"🎯 Model path specified: {args.model_path}")
        print("   Running demo with trained model...")
        quick_demo(args.model_path)
    elif args.auto_run:
        print("🚀 Auto-run mode enabled")
        print("   Running demo with untrained model...")
        quick_demo()
    else:
        # Ask user if they want to run the demo
        try:
            response = input("Run the AI agent demo? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                quick_demo()
            else:
                print("Demo skipped. You can run it later with:")
                print("  python3 quick_start.py                    # Untrained demo")
                print("  python3 quick_start.py --model-path models/suika_ai_trained.h5  # Trained demo")
        except KeyboardInterrupt:
            print("\nDemo interrupted. Goodbye!")
        except Exception as e:
            print(f"Error: {e}")
            print("Running demo anyway...")
            quick_demo() 