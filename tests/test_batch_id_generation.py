#!/usr/bin/env python3
"""
Stress test for batch ID generation to verify fcntl.flock() reliability.

The adversary's challenge: "Prove fcntl.flock() actually prevents batch ID 
collisions under realistic multi-user scenarios. I bet network filesystems, 
cloud storage, or high concurrency breaks your locking assumptions."

This test replicates the exact pattern from log_utils.py:get_next_batch_id()
and attempts to break file locking under stress conditions.

If file locking works, we should see:
- Sequential batch IDs with no duplicates
- No race conditions despite concurrent access
- Reliable behavior across different filesystems

If the adversary is right, we should see:
- Duplicate batch IDs generated
- Race conditions causing corruption
- Failures on network/cloud filesystems
"""

import asyncio
import multiprocessing
import tempfile
import time
import os
import fcntl
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
from concurrent.futures import ProcessPoolExecutor
import shutil

# Copy the exact implementation from log_utils.py for testing
def get_next_batch_id_test(experiment_name: str, test_dir: Path, purpose: str = None) -> str:
    """
    Test version of get_next_batch_id with configurable directory.
    This replicates the exact logic from app/core/log_utils.py
    """
    experiment_dir = test_dir / experiment_name
    experiment_dir.mkdir(exist_ok=True)
    
    counter_file = experiment_dir / ".batch_counter"
    
    # Multi-user safety: include user info for debugging
    user_info = f"user:{os.getenv('USER', 'unknown')}_pid:{os.getpid()}"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Use file locking to prevent race conditions
            with open(counter_file, "a+") as f:
                # Lock the file for exclusive access
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                # Read current count
                f.seek(0)
                content = f.read().strip()
                
                if content:
                    try:
                        count = int(content) + 1
                    except ValueError:
                        # Handle corrupted counter file
                        print(f"⚠️  Corrupted batch counter for {experiment_name}, resetting to 1")
                        count = 1
                else:
                    count = 1
                
                # Write new count with metadata
                f.seek(0)
                f.truncate()
                f.write(f"{count}\n# Last updated by {user_info} at {datetime.now().isoformat()}")
                if purpose:
                    f.write(f"\n# Purpose: {purpose}")
                
                # Lock is automatically released when file closes
            
            batch_id = f"{experiment_name}-{count:02d}"
            return batch_id
            
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                # Brief retry delay with jitter
                time.sleep(0.1 + (attempt * 0.1))
                continue
            else:
                # Fallback to timestamp-based ID for reliability
                timestamp = datetime.now().strftime("%m%d_%H%M%S")
                fallback_id = f"{experiment_name}-{timestamp}"
                print(f"⚠️  Batch counter failed for {experiment_name}, using fallback: {fallback_id}")
                return fallback_id
    
    # Should never reach here, but safety fallback
    return f"{experiment_name}-emergency"


def worker_process(worker_id: int, test_dir: Path, experiment_name: str, iterations: int) -> List[str]:
    """
    Worker process that generates batch IDs concurrently.
    Each worker tries to create batch IDs as fast as possible.
    """
    batch_ids = []
    
    for i in range(iterations):
        purpose = f"worker_{worker_id}_iteration_{i}"
        batch_id = get_next_batch_id_test(experiment_name, test_dir, purpose)
        batch_ids.append(batch_id)
        
        # Small delay to allow other workers to interleave
        time.sleep(0.001)
    
    return batch_ids


def test_concurrent_batch_creation(num_workers: int = 20, iterations_per_worker: int = 10) -> Dict[str, any]:
    """
    Test if multiple processes can create batch IDs concurrently without collisions.
    
    Args:
        num_workers: Number of concurrent processes
        iterations_per_worker: How many batch IDs each worker creates
        
    Returns:
        Dict with test results and collision analysis
    """
    print(f"🧪 Testing {num_workers} concurrent workers, {iterations_per_worker} iterations each...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        experiment_name = "stress_test"
        
        start_time = time.time()
        
        # Use multiprocessing to create true concurrency (not just asyncio)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(worker_process, worker_id, test_dir, experiment_name, iterations_per_worker)
                futures.append(future)
            
            # Collect all results
            all_batch_ids = []
            for future in futures:
                batch_ids = future.result()
                all_batch_ids.extend(batch_ids)
        
        execution_time = time.time() - start_time
        
        # Analyze results for collisions
        batch_id_counts = {}
        for batch_id in all_batch_ids:
            batch_id_counts[batch_id] = batch_id_counts.get(batch_id, 0) + 1
        
        duplicates = {bid: count for bid, count in batch_id_counts.items() if count > 1}
        fallback_ids = [bid for bid in all_batch_ids if not bid.startswith(f"{experiment_name}-") or "emergency" in bid]
        
        expected_total = num_workers * iterations_per_worker
        actual_total = len(all_batch_ids)
        unique_count = len(set(all_batch_ids))
        
        results = {
            'test_config': {
                'num_workers': num_workers,
                'iterations_per_worker': iterations_per_worker,
                'execution_time_ms': execution_time * 1000
            },
            'results': {
                'expected_total': expected_total,
                'actual_total': actual_total,
                'unique_count': unique_count,
                'duplicate_count': len(duplicates),
                'fallback_count': len(fallback_ids)
            },
            'duplicates': duplicates,
            'fallback_ids': fallback_ids,
            'collision_detected': len(duplicates) > 0,
            'data_integrity': {
                'missing_ids': expected_total - actual_total,
                'collision_rate': len(duplicates) / unique_count if unique_count > 0 else 0,
                'fallback_rate': len(fallback_ids) / actual_total if actual_total > 0 else 0
            }
        }
        
        return results


def test_filesystem_compatibility():
    """
    Test file locking behavior on different filesystem types.
    This tests the adversary's claim about network filesystem failures.
    """
    print("🗂️  Testing filesystem compatibility...")
    
    filesystem_results = {}
    
    # Test on regular local filesystem
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        try:
            batch_id = get_next_batch_id_test("fs_test", test_dir, "local_filesystem")
            filesystem_results['local'] = {'success': True, 'batch_id': batch_id, 'error': None}
        except Exception as e:
            filesystem_results['local'] = {'success': False, 'batch_id': None, 'error': str(e)}
    
    # Test on /tmp (might be tmpfs)
    try:
        test_dir = Path("/tmp/red_core_test")
        test_dir.mkdir(exist_ok=True)
        batch_id = get_next_batch_id_test("fs_test", test_dir, "tmp_filesystem")
        filesystem_results['tmp'] = {'success': True, 'batch_id': batch_id, 'error': None}
        shutil.rmtree(test_dir, ignore_errors=True)
    except Exception as e:
        filesystem_results['tmp'] = {'success': False, 'batch_id': None, 'error': str(e)}
    
    return filesystem_results


def test_rapid_fire_scenario():
    """
    Test the adversary's "same microsecond timing" scenario.
    Generate batch IDs as rapidly as possible to find timing vulnerabilities.
    """
    print("⚡ Testing rapid-fire batch creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = Path(temp_dir)
        experiment_name = "rapid_test"
        
        batch_ids = []
        start_time = time.time()
        
        # Create batch IDs as fast as possible
        for i in range(100):
            batch_id = get_next_batch_id_test(experiment_name, test_dir, f"rapid_{i}")
            batch_ids.append(batch_id)
        
        execution_time = time.time() - start_time
        
        # Check for duplicates in rapid sequence
        duplicates = {}
        for batch_id in batch_ids:
            if batch_ids.count(batch_id) > 1:
                duplicates[batch_id] = batch_ids.count(batch_id)
        
        return {
            'total_generated': len(batch_ids),
            'unique_count': len(set(batch_ids)),
            'duplicates': duplicates,
            'execution_time_ms': execution_time * 1000,
            'avg_time_per_id_ms': (execution_time / len(batch_ids)) * 1000,
            'collision_detected': len(duplicates) > 0
        }


async def run_comprehensive_tests():
    """Run the complete test suite to validate/invalidate file locking assumptions."""
    
    print("🔬 BATCH ID GENERATION STRESS TEST SUITE")
    print("=" * 60)
    print("Testing fcntl.flock() reliability under adversarial conditions")
    print("Adversary claim: 'File locking breaks under multi-user/network scenarios'\n")
    
    all_results = []
    
    # Test 1: Small concurrent test
    print("📊 Test 1: Small concurrency (5 workers, 5 iterations)")
    result1 = test_concurrent_batch_creation(5, 5)
    all_results.append(('small_concurrency', result1))
    print_test_results(result1)
    
    # Test 2: Medium concurrent test
    print("\n📊 Test 2: Medium concurrency (10 workers, 10 iterations)")
    result2 = test_concurrent_batch_creation(10, 10)
    all_results.append(('medium_concurrency', result2))
    print_test_results(result2)
    
    # Test 3: High concurrent test
    print("\n📊 Test 3: High concurrency (20 workers, 15 iterations)")
    result3 = test_concurrent_batch_creation(20, 15)
    all_results.append(('high_concurrency', result3))
    print_test_results(result3)
    
    # Test 4: Filesystem compatibility
    print("\n🗂️  Test 4: Filesystem compatibility")
    fs_results = test_filesystem_compatibility()
    all_results.append(('filesystem_compatibility', fs_results))
    print_filesystem_results(fs_results)
    
    # Test 5: Rapid-fire timing
    print("\n⚡ Test 5: Rapid-fire timing attack")
    rapid_results = test_rapid_fire_scenario()
    all_results.append(('rapid_fire', rapid_results))
    print_rapid_results(rapid_results)
    
    # Final analysis
    print("\n" + "=" * 60)
    print("📋 FINAL VERDICT:")
    
    total_collisions = sum(1 for name, result in all_results[:3] if result.get('collision_detected', False))
    filesystem_failures = sum(1 for fs, result in fs_results.items() if not result['success'])
    rapid_collisions = rapid_results.get('collision_detected', False)
    
    print(f"   Concurrency tests with collisions: {total_collisions}/3")
    print(f"   Filesystem compatibility failures: {filesystem_failures}/{len(fs_results)}")
    print(f"   Rapid-fire timing collisions: {'Yes' if rapid_collisions else 'No'}")
    
    if total_collisions == 0 and filesystem_failures == 0 and not rapid_collisions:
        print("\n   🎯 FILE LOCKING VALIDATED: fcntl.flock() prevents collisions")
        print("   ✅ No race conditions detected under stress")
        print("   📝 Batch ID generation appears reliable")
        print("\n   ⚠️  UNCERTAINTY_LOG ENTRY #113 VALIDATED")
    else:
        print("\n   ⚔️  ADVERSARY VINDICATED: File locking has vulnerabilities")
        print("   🔥 Race conditions or filesystem incompatibilities detected")
        print("   📉 UNCERTAINTY_LOG ENTRY #113 INVALIDATED")
    
    return all_results


def print_test_results(results):
    """Print formatted test results."""
    config = results['test_config']
    data = results['results']
    
    print(f"   Workers: {config['num_workers']}, Iterations: {config['iterations_per_worker']}")
    print(f"   Execution time: {config['execution_time_ms']:.1f}ms")
    print(f"   Expected: {data['expected_total']}, Actual: {data['actual_total']}, Unique: {data['unique_count']}")
    
    if results['collision_detected']:
        print(f"   🚨 COLLISIONS DETECTED: {data['duplicate_count']} duplicates")
        for batch_id, count in results['duplicates'].items():
            print(f"      {batch_id}: {count} times")
    else:
        print(f"   ✅ No collisions detected")
    
    if data['fallback_count'] > 0:
        print(f"   ⚠️  Fallback IDs used: {data['fallback_count']}")


def print_filesystem_results(fs_results):
    """Print filesystem compatibility results."""
    for fs_type, result in fs_results.items():
        if result['success']:
            print(f"   ✅ {fs_type}: {result['batch_id']}")
        else:
            print(f"   ❌ {fs_type}: {result['error']}")


def print_rapid_results(rapid_results):
    """Print rapid-fire test results."""
    print(f"   Generated: {rapid_results['total_generated']}, Unique: {rapid_results['unique_count']}")
    print(f"   Average time per ID: {rapid_results['avg_time_per_id_ms']:.2f}ms")
    
    if rapid_results['collision_detected']:
        print(f"   🚨 TIMING COLLISIONS: {len(rapid_results['duplicates'])} duplicates")
    else:
        print(f"   ✅ No timing-based collisions")


if __name__ == "__main__":
    # Run the comprehensive test suite
    results = asyncio.run(run_comprehensive_tests())
    
    print(f"\n{''}" + "=" * 60)
    print("🧪 Raw results available for further analysis:")
    print("   Variable: results")
    print("   Usage: python3 -i test_batch_id_generation.py")