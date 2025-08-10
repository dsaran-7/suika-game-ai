#!/usr/bin/env python3
"""
Basic Suika Environment Test
Tests the basic environment functionality without requiring TensorFlow.
"""

import gymnasium
import suika_env
import numpy as np
import time

def test_basic_environment():
    """Test the basic Suika environment without AI agent."""
    
    print("🧪 Testing Basic Suika Environment")
    print("=" * 40)
    
    try:
        # Test basic environment creation
        print("✅ Creating basic environment...")
        env = gymnasium.make("SuikaEnv-v0")
        print("✅ Environment created successfully!")
        
        # Test reset
        print("✅ Testing environment reset...")
        obs, info = env.reset()
        print(f"✅ Reset successful! Observation keys: {list(obs.keys())}")
        
        # Test observation space
        if 'image' in obs:
            print(f"✅ Image shape: {obs['image'].shape}")
        if 'score' in obs:
            print(f"✅ Score: {obs['score']}")
        
        # Test action space
        print(f"✅ Action space: {env.action_space}")
        print(f"✅ Observation space: {env.observation_space}")
        
        # Test a few random steps
        print("✅ Testing random actions...")
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"  Step {step + 1}: Action {action[0]:.3f}, Reward {reward:.1f}, Score {obs['score'].item():.0f}")
            
            if terminated:
                print("  ✅ Episode terminated normally")
                break
        
        # Test environment close
        print("✅ Testing environment close...")
        env.close()
        print("✅ Environment closed successfully!")
        
        print("\n🎉 Basic environment test PASSED!")
        print("   The core Suika environment is working correctly.")
        print("   You can now proceed to test the AI agent features.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Basic environment test FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_imports():
    """Test if all required modules can be imported."""
    
    print("📦 Testing Module Imports")
    print("=" * 30)
    
    modules_to_test = [
        ('gymnasium', 'gymnasium'),
        ('numpy', 'np'),
        ('PIL', 'PIL'),
        ('selenium', 'selenium'),
        ('imageio', 'imageio'),
    ]
    
    all_imports_ok = True
    
    for module_name, import_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ {module_name} import failed: {e}")
            all_imports_ok = False
    
    # Test TensorFlow separately
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow imported successfully (version: {tf.__version__})")
    except ImportError as e:
        print(f"⚠️  TensorFlow import failed: {e}")
        print("   This is expected if TensorFlow isn't installed yet.")
        print("   The basic environment will still work without it.")
    
    return all_imports_ok

def main():
    """Main test function."""
    
    print("🍉 Suika Environment Basic Test")
    print("=" * 40)
    print("This script tests the basic environment functionality")
    print("without requiring the full AI agent setup.")
    print()
    
    # Test imports first
    imports_ok = test_imports()
    print()
    
    if imports_ok:
        # Test basic environment
        env_ok = test_basic_environment()
        
        if env_ok:
            print("\n" + "=" * 50)
            print("🎉 ALL BASIC TESTS PASSED!")
            print("=" * 50)
            print("Your Suika environment is working correctly!")
            print("\nNext steps:")
            print("1. Install TensorFlow: pip3 install tensorflow-macos")
            print("2. Test AI agent: python3 quick_start.py")
            print("3. Run training: python3 train_ai_agent.py")
        else:
            print("\n❌ Basic environment test failed.")
            print("Please check the error messages above.")
    else:
        print("\n❌ Some required modules are missing.")
        print("Please install missing dependencies and try again.")

if __name__ == "__main__":
    main() 