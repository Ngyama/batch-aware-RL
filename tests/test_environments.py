"""
Quick Test Script for Both Environments

This script runs a quick sanity check on both simulation and real environments
to ensure they are working correctly before training.

Usage:
    python test_environments.py
"""

import numpy as np
from src.environment_sim import SchedulingEnvSim
from src.environment_real import SchedulingEnvReal
import src.constants as c


def test_simulation_env():
    """Test the simulation environment."""
    print("="*70)
    print("🧪 Testing Simulation Environment (方案A)")
    print("="*70 + "\n")
    
    try:
        # Create environment
        print("1. Creating environment...")
        env = SchedulingEnvSim()
        print("   ✅ Environment created\n")
        
        # Test reset
        print("2. Testing reset()...")
        state, info = env.reset()
        print(f"   ✅ Initial state: {state}")
        print(f"   ✅ State shape: {state.shape}")
        assert state.shape == (c.NUM_STATE_FEATURES,), "State shape mismatch!"
        print()
        
        # Test action space
        print("3. Testing action space...")
        print(f"   ✅ Action space: Discrete({env.action_space.n})")
        assert env.action_space.n == c.NUM_ACTIONS, "Action space size mismatch!"
        print()
        
        # Test a few steps
        print("4. Running 20 test steps...")
        for i in range(20):
            # Random action
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"   Episode ended at step {i+1}")
                state, info = env.reset()
            
            if (i + 1) % 5 == 0:
                print(f"   Step {i+1}: State={state}, Reward={reward:.2f}")
        
        print("   ✅ Steps executed successfully\n")
        
        # Test render
        print("5. Testing render()...")
        env.render()
        print()
        
        # Close
        env.close()
        print("✅ SIMULATION ENVIRONMENT TEST PASSED!\n")
        return True
        
    except Exception as e:
        print(f"❌ SIMULATION ENVIRONMENT TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_real_env():
    """Test the real data environment."""
    print("="*70)
    print("🧪 Testing Real Data Environment (方案B)")
    print("="*70 + "\n")
    
    print("⚠️  This test requires:")
    print("   - Imagenette dataset downloaded")
    print("   - PyTorch with CUDA (recommended) or CPU")
    print()
    
    try:
        # Create environment
        print("1. Creating environment (this may take a moment)...")
        env = SchedulingEnvReal()
        print("   ✅ Environment created\n")
        
        # Test reset
        print("2. Testing reset()...")
        state, info = env.reset()
        print(f"   ✅ Initial state: {state}")
        print(f"   ✅ State shape: {state.shape}")
        assert state.shape == (c.NUM_STATE_FEATURES,), "State shape mismatch!"
        print()
        
        # Test action space
        print("3. Testing action space...")
        print(f"   ✅ Action space: Discrete({env.action_space.n})")
        assert env.action_space.n == c.NUM_ACTIONS, "Action space size mismatch!"
        print()
        
        # Test a few steps (fewer than sim because it's slower)
        print("4. Running 10 test steps with real inference...")
        for i in range(10):
            # Test different actions
            action = i % env.action_space.n
            state, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"   Episode ended at step {i+1}")
                state, info = env.reset()
            
            if (i + 1) % 5 == 0:
                print(f"   Step {i+1}: State={state}, Reward={reward:.2f}")
        
        print("   ✅ Steps with real inference executed successfully\n")
        
        # Test render
        print("5. Testing render()...")
        env.render()
        print()
        
        # Close
        env.close()
        print("✅ REAL ENVIRONMENT TEST PASSED!\n")
        return True
        
    except FileNotFoundError as e:
        print(f"❌ Dataset not found: {e}")
        print("   Please run: python scripts/download_dataset.py\n")
        return False
    
    except Exception as e:
        print(f"❌ REAL ENVIRONMENT TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("🚀 BATCH-AWARE RL SCHEDULER - ENVIRONMENT TESTS")
    print("="*70 + "\n")
    
    # Test simulation environment
    sim_passed = test_simulation_env()
    
    # Ask before testing real environment (it's slower)
    print("="*70)
    print("The real environment test will:")
    print("  - Load ResNet-18 model (~50MB)")
    print("  - Load Imagenette dataset")
    print("  - Run actual neural network inference")
    print("  - Take 1-2 minutes to complete")
    print("="*70 + "\n")
    
    response = input("Do you want to test the real environment? (y/n): ")
    
    if response.lower() == 'y':
        print()
        real_passed = test_real_env()
    else:
        print("\n⏭️  Skipping real environment test.\n")
        real_passed = None
    
    # Summary
    print("="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"  Simulation Environment: {'✅ PASSED' if sim_passed else '❌ FAILED'}")
    if real_passed is not None:
        print(f"  Real Environment: {'✅ PASSED' if real_passed else '❌ FAILED'}")
    else:
        print(f"  Real Environment: ⏭️  SKIPPED")
    print("="*70 + "\n")
    
    if sim_passed and (real_passed is None or real_passed):
        print("🎉 All tests passed! You're ready to start training.")
        print("\nNext steps:")
        print("  1. For quick experimentation: python train_sim.py")
        print("  2. For realistic training: python train_real.py")
        print("  3. Read README_BATCH_AWARE.md for detailed guide")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()

