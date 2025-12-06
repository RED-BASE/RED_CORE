#!/usr/bin/env python3
"""
Test script to verify if asyncio.gather creates observable race conditions
when modifying shared state, as claimed in UNCERTAINTY_LOG.md entry #001.

The adversary's challenge: "I bet asyncio's GIL protection or event loop 
scheduling makes your 'race conditions' impossible."

This test replicates the exact pattern from run_experiments.py:
- Multiple async tasks modifying shared variables
- Lists being appended to concurrently
- Counters being incremented

If race conditions exist, we should see:
- Missing entries in lists
- Incorrect final counter values
- Data corruption

If the adversary is right, we should see:
- All operations complete atomically
- Perfect data integrity
- No observable corruption
"""

import asyncio
import time
from typing import List, Dict

# Shared state that mimics run_experiments.py patterns
turn_counter = 0
successes: List[tuple] = []
failures: List[tuple] = []
model_failure_counts: Dict[str, int] = {}

async def simulate_experiment_task(task_id: int, model_name: str) -> None:
    """
    Simulate a single experiment task that modifies shared state.
    This replicates the exact patterns from run_experiments.py lines 508, 533, 538, 539
    """
    global turn_counter
    
    # Simulate some async work (like API calls)
    await asyncio.sleep(0.001)  # Tiny delay to allow task interleaving
    
    # Pattern from line 508: turn_counter += 1
    turn_counter += 1
    
    # Simulate success or failure randomly
    if task_id % 7 == 0:  # Some tasks "fail"
        # Pattern from line 538: failures.append(...)
        failures.append((f"{model_name}({task_id})", f"error_{task_id}", f"traceback_{task_id}"))
        
        # Pattern from line 539: model_failure_counts[model] = model_failure_counts.get(model, 0) + 1
        model_failure_counts[model_name] = model_failure_counts.get(model_name, 0) + 1
    else:
        # Pattern from line 533: successes.append(...)
        successes.append((f"{model_name}({task_id})", f"run_id_{task_id}", f"log_path_{task_id}"))

async def test_race_conditions(num_tasks: int = 1000, num_models: int = 5) -> Dict[str, any]:
    """
    Test if asyncio.gather creates race conditions with shared state modification.
    
    Args:
        num_tasks: Number of concurrent tasks to run
        num_models: Number of different model names to simulate
        
    Returns:
        Dict with test results and integrity checks
    """
    global turn_counter, successes, failures, model_failure_counts
    
    # Reset shared state
    turn_counter = 0
    successes.clear()
    failures.clear()
    model_failure_counts.clear()
    
    print(f"🧪 Testing {num_tasks} concurrent tasks across {num_models} models...")
    
    # Create tasks that replicate run_experiments.py patterns
    model_names = [f"model_{i}" for i in range(num_models)]
    tasks = []
    
    for task_id in range(num_tasks):
        model_name = model_names[task_id % num_models]
        task = simulate_experiment_task(task_id, model_name)
        tasks.append(task)
    
    # This is the exact pattern from run_experiments.py line 545
    start_time = time.time()
    await asyncio.gather(*tasks, return_exceptions=True)
    execution_time = time.time() - start_time
    
    # Calculate expected vs actual values
    expected_failures = sum(1 for i in range(num_tasks) if i % 7 == 0)
    expected_successes = num_tasks - expected_failures
    expected_turn_counter = num_tasks
    
    actual_failures = len(failures)
    actual_successes = len(successes)
    actual_turn_counter = turn_counter
    
    # Check for data corruption
    corruption_detected = (
        actual_failures != expected_failures or
        actual_successes != expected_successes or
        actual_turn_counter != expected_turn_counter
    )
    
    results = {
        'test_config': {
            'num_tasks': num_tasks,
            'num_models': num_models,
            'execution_time_ms': execution_time * 1000
        },
        'expected': {
            'turn_counter': expected_turn_counter,
            'successes': expected_successes,
            'failures': expected_failures
        },
        'actual': {
            'turn_counter': actual_turn_counter,
            'successes': actual_successes,
            'failures': actual_failures
        },
        'corruption_detected': corruption_detected,
        'data_integrity': {
            'missing_successes': expected_successes - actual_successes,
            'missing_failures': expected_failures - actual_failures,
            'counter_drift': expected_turn_counter - actual_turn_counter
        },
        'failure_counts': dict(model_failure_counts)
    }
    
    return results

async def run_multiple_tests():
    """Run multiple test scenarios to thoroughly check for race conditions."""
    
    test_scenarios = [
        (100, 3),    # Small test
        (1000, 5),   # Medium test  
        (5000, 10),  # Large test
        (10000, 20)  # Stress test
    ]
    
    print("🔬 ASYNC RACE CONDITION TEST SUITE")
    print("=" * 50)
    print("Testing if asyncio.gather creates observable data corruption")
    print("when multiple tasks modify shared variables concurrently.\n")
    
    all_results = []
    
    for num_tasks, num_models in test_scenarios:
        results = await test_race_conditions(num_tasks, num_models)
        all_results.append(results)
        
        print(f"📊 Test: {num_tasks} tasks, {num_models} models")
        print(f"   Execution time: {results['test_config']['execution_time_ms']:.1f}ms")
        print(f"   Expected: counter={results['expected']['turn_counter']}, "
              f"successes={results['expected']['successes']}, "
              f"failures={results['expected']['failures']}")
        print(f"   Actual:   counter={results['actual']['turn_counter']}, "
              f"successes={results['actual']['successes']}, "
              f"failures={results['actual']['failures']}")
        
        if results['corruption_detected']:
            print(f"   🚨 CORRUPTION DETECTED!")
            print(f"      Missing successes: {results['data_integrity']['missing_successes']}")
            print(f"      Missing failures: {results['data_integrity']['missing_failures']}")
            print(f"      Counter drift: {results['data_integrity']['counter_drift']}")
        else:
            print(f"   ✅ No corruption detected")
        
        print()
    
    # Summary analysis
    corrupted_tests = sum(1 for r in all_results if r['corruption_detected'])
    total_tests = len(all_results)
    
    print("=" * 50)
    print("📋 FINAL VERDICT:")
    print(f"   Tests with corruption: {corrupted_tests}/{total_tests}")
    
    if corrupted_tests == 0:
        print("   🎯 ADVERSARY WAS RIGHT: No race conditions detected")
        print("   🔍 asyncio.gather appears to maintain data integrity")
        print("   📝 Python's GIL and event loop scheduling prevent corruption")
        print("\n   ⚠️  UNCERTAINTY_LOG ENTRY #001 INVALIDATED")
        print("   📉 All downstream analysis based on race conditions is CONTAMINATED")
    else:
        print("   ⚔️  RACE CONDITIONS CONFIRMED: Data corruption observed")
        print("   🔥 Shared state modification in asyncio.gather is unsafe")
        print("   📈 UNCERTAINTY_LOG ENTRY #001 VALIDATED")
    
    return all_results

if __name__ == "__main__":
    # Run the test suite
    results = asyncio.run(run_multiple_tests())
    
    print("\n" + "=" * 50)
    print("🧪 Raw results available for further analysis:")
    print("   Variable: results")
    print("   Usage: python3 -i test_async_race_conditions.py")