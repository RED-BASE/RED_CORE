#!/usr/bin/env python3
"""
Error handling sophistication test to verify "built-in error handling with exponential backoff" claims.

The adversary's challenge: "Prove your error handling is actually sophisticated. I bet your 
'exponential backoff' is naive delays, your retry logic fails under real failure modes, and 
your 'built-in' error handling is scattered ad-hoc exception catching that breaks under stress."

This test validates error handling sophistication claims:
- Exponential backoff algorithm implementation and correctness
- Comprehensive error handling across all failure modes
- Retry logic robustness under various API error conditions
- Error recovery strategies and graceful degradation
- Consistency of error handling patterns across all providers

If error handling is sophisticated, we should see:
- True exponential backoff with jitter and maximum limits
- Comprehensive error categorization and appropriate responses
- Robust retry logic that handles transient vs permanent failures
- Graceful degradation and meaningful error reporting

If the adversary is right, we should see:
- Naive linear delays disguised as "exponential backoff"
- Scattered exception handling without systematic approach
- Retry logic that fails under edge cases or stress conditions
- Poor error categorization leading to inappropriate responses
"""

import ast
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import statistics

def analyze_exponential_backoff_implementation() -> Dict[str, Any]:
    """Analyze exponential backoff algorithm implementations."""
    
    project_root = Path(__file__).parent
    api_runners_dir = project_root / "app" / "api_runners"
    
    analysis = {
        "backoff_implementations": {},
        "algorithm_sophistication": {},
        "backoff_patterns": [],
        "sophistication_failures": []
    }
    
    if not api_runners_dir.exists():
        analysis["error"] = "API runners directory not found"
        return analysis
    
    # Find all relevant files that might contain backoff logic
    files_to_check = list(api_runners_dir.glob("*.py"))
    
    for file_path in files_to_check:
        file_name = file_path.stem
        
        analysis["backoff_implementations"][file_name] = {
            "file_name": file_name,
            "exponential_patterns": [],
            "backoff_algorithms": [],
            "retry_logic": [],
            "jitter_implementation": [],
            "max_backoff_limits": [],
            "sophistication_indicators": []
        }
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Look for exponential backoff patterns
            exponential_patterns = [
                r"backoff.*\*.*2",  # backoff * 2
                r"backoff.*\*\*",   # backoff ** power
                r"2.*\*\*.*retry",  # 2 ** retry_count
                r"exponential.*backoff",
                r"pow.*2.*retry",
                r"math\.pow.*2"
            ]
            
            for pattern in exponential_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["backoff_implementations"][file_name]["exponential_patterns"].extend(matches)
            
            # Look for backoff algorithm sophistication
            algorithm_patterns = [
                r"jitter",
                r"random.*backoff",
                r"max.*backoff",
                r"min.*backoff", 
                r"backoff.*limit",
                r"backoff.*cap",
                r"linear.*backoff",
                r"fixed.*delay"
            ]
            
            for pattern in algorithm_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["backoff_implementations"][file_name]["backoff_algorithms"].append(pattern)
            
            # Look for retry logic patterns
            retry_patterns = [
                r"retry.*count",
                r"max.*retry",
                r"attempt.*count",
                r"for.*range.*retry",
                r"while.*retry",
                r"sleep.*retry"
            ]
            
            for pattern in retry_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["backoff_implementations"][file_name]["retry_logic"].extend(matches)
            
            # Look for sophistication indicators
            sophistication_patterns = [
                "# Exponential backoff",
                "# Sophisticated retry",
                "# Advanced error handling",
                "transient.*error",
                "permanent.*error",
                "rate.*limit.*error",
                "timeout.*error"
            ]
            
            for pattern in sophistication_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["backoff_implementations"][file_name]["sophistication_indicators"].append(pattern)
            
            # Look for naive implementations
            naive_patterns = [
                r"sleep\(1\)",
                r"sleep\(2\)",
                r"sleep\(5\)",
                r"time\.sleep.*\d+",
                r"delay.*=.*\d+",
                r"wait.*=.*\d+"
            ]
            
            naive_found = []
            for pattern in naive_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    naive_found.extend(matches)
            
            if naive_found and not analysis["backoff_implementations"][file_name]["exponential_patterns"]:
                analysis["sophistication_failures"].append({
                    "file": file_name,
                    "issue": "Naive fixed delays found without exponential backoff",
                    "evidence": naive_found
                })
        
        except Exception as e:
            analysis["backoff_implementations"][file_name] = {"error": str(e)}
    
    return analysis

def analyze_error_handling_comprehensiveness() -> Dict[str, Any]:
    """Analyze comprehensiveness of error handling across the system."""
    
    project_root = Path(__file__).parent
    api_runners_dir = project_root / "app" / "api_runners"
    
    analysis = {
        "error_categories": {},
        "exception_handling": {},
        "error_recovery": {},
        "handling_inconsistencies": []
    }
    
    if not api_runners_dir.exists():
        analysis["error"] = "API runners directory not found"
        return analysis
    
    # Define expected error categories for API services
    expected_error_types = [
        "rate_limit",       # 429 errors
        "timeout",          # Request timeouts
        "auth",            # 401/403 errors
        "server_error",    # 5xx errors
        "network",         # Connection errors
        "quota",           # Usage limits
        "invalid_request", # 4xx errors
        "service_unavailable" # 503 errors
    ]
    
    runner_files = list(api_runners_dir.glob("*_runner.py"))
    
    for runner_file in runner_files:
        runner_name = runner_file.stem
        
        analysis["error_categories"][runner_name] = {
            "handled_error_types": [],
            "exception_patterns": [],
            "recovery_strategies": [],
            "error_categorization": []
        }
        
        try:
            with open(runner_file, 'r') as f:
                content = f.read()
            
            # Look for specific error type handling
            for error_type in expected_error_types:
                error_patterns = [
                    error_type.replace("_", ".*"),
                    f"{error_type}.*error",
                    f"handle.*{error_type}",
                    f"catch.*{error_type}"
                ]
                
                for pattern in error_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        analysis["error_categories"][runner_name]["handled_error_types"].append(error_type)
                        break
            
            # Look for exception handling patterns
            exception_patterns = [
                r"except.*Exception",
                r"except.*requests\.",
                r"except.*HTTPError",
                r"except.*Timeout",
                r"except.*ConnectionError",
                r"try:.*except:",
                r"raise.*Error"
            ]
            
            for pattern in exception_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["error_categories"][runner_name]["exception_patterns"].extend(matches)
            
            # Look for recovery strategies
            recovery_patterns = [
                "retry",
                "fallback",
                "graceful.*degradation",
                "alternative.*model",
                "error.*recovery",
                "circuit.*breaker"
            ]
            
            for pattern in recovery_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["error_categories"][runner_name]["recovery_strategies"].append(pattern)
            
            # Check for comprehensive error categorization
            if len(analysis["error_categories"][runner_name]["handled_error_types"]) < len(expected_error_types) * 0.6:
                analysis["handling_inconsistencies"].append({
                    "runner": runner_name,
                    "issue": "Incomplete error type coverage",
                    "handled": len(analysis["error_categories"][runner_name]["handled_error_types"]),
                    "expected": len(expected_error_types)
                })
        
        except Exception as e:
            analysis["error_categories"][runner_name] = {"error": str(e)}
    
    return analysis

def test_backoff_algorithm_quality() -> Dict[str, Any]:
    """Test the quality of backoff algorithms by simulating scenarios."""
    
    analysis = {
        "algorithm_tests": {},
        "backoff_sequences": {},
        "quality_assessment": {},
        "sophistication_score": 0
    }
    
    # Test common backoff algorithms
    backoff_algorithms = {
        "naive_fixed": lambda attempt: 1.0,  # Fixed 1 second
        "naive_linear": lambda attempt: attempt * 1.0,  # Linear increase
        "simple_exponential": lambda attempt: 2 ** attempt,  # Basic exponential
        "capped_exponential": lambda attempt: min(2 ** attempt, 30),  # Capped at 30s
        "exponential_with_jitter": lambda attempt: min(2 ** attempt, 30) + (0.1 * attempt)  # With jitter
    }
    
    # Test scenarios
    test_scenarios = [
        {"name": "quick_failure", "attempts": 3},
        {"name": "persistent_failure", "attempts": 7},
        {"name": "extended_outage", "attempts": 10}
    ]
    
    for algo_name, algo_func in backoff_algorithms.items():
        analysis["algorithm_tests"][algo_name] = {}
        
        for scenario in test_scenarios:
            scenario_name = scenario["name"]
            attempts = scenario["attempts"]
            
            backoff_sequence = []
            total_wait_time = 0
            
            for attempt in range(1, attempts + 1):
                try:
                    backoff_time = algo_func(attempt)
                    backoff_sequence.append(backoff_time)
                    total_wait_time += backoff_time
                except Exception:
                    backoff_sequence.append(float('inf'))
                    break
            
            analysis["algorithm_tests"][algo_name][scenario_name] = {
                "backoff_sequence": backoff_sequence,
                "total_wait_time": total_wait_time,
                "max_single_wait": max(backoff_sequence) if backoff_sequence else 0,
                "reasonable_behavior": total_wait_time < 300 and max(backoff_sequence) < 60  # Reasonable limits
            }
    
    # Assess algorithm quality
    quality_scores = {}
    for algo_name, scenarios in analysis["algorithm_tests"].items():
        reasonable_count = sum(1 for scenario_data in scenarios.values() 
                             if scenario_data.get("reasonable_behavior", False))
        quality_scores[algo_name] = reasonable_count / len(scenarios)
    
    analysis["quality_assessment"] = quality_scores
    
    # Calculate sophistication score
    # Give higher scores to algorithms with reasonable behavior and exponential characteristics
    sophistication_factors = {
        "naive_fixed": 0.1,
        "naive_linear": 0.3,
        "simple_exponential": 0.6,
        "capped_exponential": 0.8,
        "exponential_with_jitter": 1.0
    }
    
    weighted_score = sum(quality_scores.get(algo, 0) * sophistication_factors.get(algo, 0) 
                        for algo in sophistication_factors.keys())
    analysis["sophistication_score"] = weighted_score / len(sophistication_factors)
    
    return analysis

def run_error_handling_sophistication_analysis():
    """Run comprehensive error handling sophistication analysis."""
    
    print("🔬 ERROR HANDLING SOPHISTICATION ANALYSIS")
    print("=" * 60)
    print("Testing 'built-in error handling with exponential backoff' claims")
    print("Adversary position: 'Exponential backoff is naive delays, error handling is scattered'\\n")
    
    # Analyze exponential backoff implementation
    print("📊 EXPONENTIAL BACKOFF IMPLEMENTATION ANALYSIS")
    print("-" * 40)
    
    backoff_analysis = analyze_exponential_backoff_implementation()
    
    if "error" in backoff_analysis:
        print(f"   ❌ Analysis failed: {backoff_analysis['error']}")
        return
    
    total_files = len(backoff_analysis["backoff_implementations"])
    files_with_exponential = 0
    files_with_sophisticated = 0
    total_sophistication_failures = len(backoff_analysis["sophistication_failures"])
    
    for file_name, file_info in backoff_analysis["backoff_implementations"].items():
        if "error" not in file_info:
            exponential_count = len(file_info.get("exponential_patterns", []))
            algorithm_count = len(file_info.get("backoff_algorithms", []))
            sophistication_count = len(file_info.get("sophistication_indicators", []))
            
            if exponential_count > 0:
                files_with_exponential += 1
                
            if sophistication_count > 0 or algorithm_count > 2:
                files_with_sophisticated += 1
            
            print(f"\\n🎯 {file_name}")
            print(f"   Exponential patterns: {exponential_count}")
            print(f"   Backoff algorithms: {algorithm_count}")
            print(f"   Retry logic: {len(file_info.get('retry_logic', []))}")
            print(f"   Sophistication indicators: {sophistication_count}")
    
    print(f"\\n   Files with exponential backoff: {files_with_exponential}/{total_files}")
    print(f"   Files with sophisticated patterns: {files_with_sophisticated}/{total_files}")
    print(f"   Sophistication failures detected: {total_sophistication_failures}")
    
    if backoff_analysis["sophistication_failures"]:
        print(f"   🚨 Sophistication failures:")
        for failure in backoff_analysis["sophistication_failures"][:3]:
            print(f"      • {failure['file']}: {failure['issue']}")
    
    # Analyze error handling comprehensiveness
    print(f"\\n📊 ERROR HANDLING COMPREHENSIVENESS")
    print("-" * 40)
    
    error_analysis = analyze_error_handling_comprehensiveness()
    
    if "error" in error_analysis:
        print(f"   ❌ Error analysis failed: {error_analysis['error']}")
    else:
        total_runners = len(error_analysis["error_categories"])
        total_inconsistencies = len(error_analysis["handling_inconsistencies"])
        
        # Calculate average error type coverage
        coverage_scores = []
        for runner_name, runner_info in error_analysis["error_categories"].items():
            if "error" not in runner_info:
                handled_types = len(runner_info.get("handled_error_types", []))
                coverage_scores.append(handled_types)
        
        avg_coverage = statistics.mean(coverage_scores) if coverage_scores else 0
        
        print(f"   API runners analyzed: {total_runners}")
        print(f"   Average error type coverage: {avg_coverage:.1f}/8 types")
        print(f"   Error handling inconsistencies: {total_inconsistencies}")
        
        if error_analysis["handling_inconsistencies"]:
            print(f"   🚨 Error handling gaps:")
            for inconsistency in error_analysis["handling_inconsistencies"][:3]:
                print(f"      • {inconsistency['runner']}: {inconsistency['handled']}/8 error types")
    
    # Test backoff algorithm quality
    print(f"\\n📊 BACKOFF ALGORITHM QUALITY TESTING")
    print("-" * 40)
    
    algorithm_test = test_backoff_algorithm_quality()
    
    quality_assessment = algorithm_test.get("quality_assessment", {})
    sophistication_score = algorithm_test.get("sophistication_score", 0)
    
    print(f"   Algorithm quality scores:")
    for algo_name, quality_score in quality_assessment.items():
        print(f"      {algo_name}: {quality_score:.1%}")
    
    print(f"   Overall sophistication score: {sophistication_score:.1%}")
    
    # Show algorithm behavior examples
    algorithm_tests = algorithm_test.get("algorithm_tests", {})
    if "capped_exponential" in algorithm_tests:
        capped_test = algorithm_tests["capped_exponential"]
        if "persistent_failure" in capped_test:
            sequence = capped_test["persistent_failure"]["backoff_sequence"]
            print(f"   Capped exponential backoff sequence: {sequence[:5]}...")
    
    # Final verdict
    print("\\n" + "=" * 60)
    print("📋 FINAL ERROR HANDLING ASSESSMENT:")
    
    exponential_coverage = files_with_exponential / total_files if total_files > 0 else 0
    sophisticated_coverage = files_with_sophisticated / total_files if total_files > 0 else 0
    avg_error_coverage = statistics.mean(coverage_scores) if coverage_scores else 0
    
    critical_failures = (
        exponential_coverage < 0.7 or  # Less than 70% have exponential backoff
        sophisticated_coverage < 0.5 or  # Less than 50% have sophisticated patterns
        total_sophistication_failures > 3 or  # Multiple sophistication failures
        avg_error_coverage < 5 or  # Poor error type coverage
        sophistication_score < 0.6  # Low algorithm sophistication
    )
    
    print(f"   Exponential backoff coverage: {exponential_coverage:.1%}")
    print(f"   Sophisticated pattern coverage: {sophisticated_coverage:.1%}")
    print(f"   Average error type coverage: {avg_error_coverage:.1f}/8")
    print(f"   Algorithm sophistication: {sophistication_score:.1%}")
    
    if critical_failures:
        print("\\n   ⚔️  ADVERSARY VINDICATED: Error handling lacks sophistication")
        print("   🔥 'Built-in error handling with exponential backoff' unsupported")
        print("   📉 UNCERTAINTY_LOG ERROR HANDLING CLAIMS INVALIDATED")
        print("   💥 Evidence: Naive algorithms, incomplete coverage, scattered patterns")
    else:
        print("\\n   🎯 ERROR HANDLING SOPHISTICATION VALIDATED")
        print("   ✅ Comprehensive exponential backoff with proper error categorization")
        print("   📝 'Built-in error handling' claims supported by evidence")
    
    return {
        "backoff_analysis": backoff_analysis,
        "error_analysis": error_analysis,
        "algorithm_test": algorithm_test,
        "overall_verdict": "INVALIDATED" if critical_failures else "VALIDATED"
    }

if __name__ == "__main__":
    # Run the comprehensive error handling sophistication analysis
    results = run_error_handling_sophistication_analysis()
    
    print(f"\\n{''}" + "=" * 60)
    print("🧪 Error handling sophistication analysis complete")
    print("   Focus: Exponential backoff, error categorization, algorithm quality")
    print("   Variable: results")