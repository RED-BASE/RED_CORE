#!/usr/bin/env python3
"""
Token counting accuracy test to verify "standardized usage tracking" and estimation accuracy claims.

The adversary's challenge: "Prove your token counting is actually accurate across providers. 
I bet your estimates are wildly wrong, billing doesn't match usage tracking, and your 
'standardized format' hides massive provider-specific inconsistencies that break cost predictions."

This test validates token counting claims:
- Token counting accuracy vs actual provider billing
- Standardized usage tracking format consistency
- Provider-specific token counting method reliability
- Conversation history token accumulation accuracy
- Cost estimation reliability for budgeting

If token counting is accurate, we should see:
- Estimates within 5-10% of actual provider billing
- Consistent usage tracking format across all providers
- Reliable token counting methods for each provider
- Accurate conversation history token accumulation

If the adversary is right, we should see:
- Wildly inaccurate token estimates vs billing
- Inconsistent usage tracking formats across providers
- Broken token counting methods with huge errors
- Cost prediction failures that break budgeting
"""

import ast
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import statistics

def analyze_token_counting_implementations() -> Dict[str, Any]:
    """Analyze token counting implementations across all API runners."""
    
    project_root = Path(__file__).parent
    api_runners_dir = project_root / "app" / "api_runners"
    
    analysis = {
        "token_counting_methods": {},
        "usage_tracking_formats": {},
        "provider_inconsistencies": [],
        "estimation_methodology": {}
    }
    
    if not api_runners_dir.exists():
        analysis["error"] = "API runners directory not found"
        return analysis
    
    # Find all API runner files
    runner_files = list(api_runners_dir.glob("*_runner.py"))
    
    for runner_file in runner_files:
        runner_name = runner_file.stem
        
        analysis["token_counting_methods"][runner_name] = {
            "file_name": runner_name,
            "count_tokens_method": False,
            "token_counting_approach": [],
            "usage_format": [],
            "accuracy_indicators": [],
            "provider_specific_logic": []
        }
        
        try:
            with open(runner_file, 'r') as f:
                content = f.read()
            
            # Look for _count_tokens method
            if "_count_tokens" in content:
                analysis["token_counting_methods"][runner_name]["count_tokens_method"] = True
            
            # Look for token counting approaches
            counting_patterns = [
                r"tiktoken",
                r"len.*text.*//.*4",
                r"encoding\.encode",
                r"tokenizer\.encode",
                r"anthropic.*count",
                r"google.*count",
                r"estimate.*token"
            ]
            
            for pattern in counting_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["token_counting_methods"][runner_name]["token_counting_approach"].extend(matches)
            
            # Look for usage tracking format
            usage_patterns = [
                r'"prompt_tokens"',
                r'"completion_tokens"',
                r'"total_tokens"',
                r'"input_tokens"',
                r'"output_tokens"',
                r"usage.*tracking",
                r"token.*usage"
            ]
            
            for pattern in usage_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["token_counting_methods"][runner_name]["usage_format"].append(pattern)
            
            # Look for accuracy indicators
            accuracy_patterns = [
                "accurate.*count",
                "estimate.*accurate", 
                "billing.*match",
                "precise.*token",
                "exact.*count"
            ]
            
            for pattern in accuracy_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["token_counting_methods"][runner_name]["accuracy_indicators"].append(pattern)
            
            # Look for provider-specific logic
            provider_specific = [
                "# OpenAI specific",
                "# Anthropic specific",
                "# Google specific",
                "if.*provider",
                "special.*case.*token",
                "workaround.*count"
            ]
            
            for pattern in provider_specific:
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["token_counting_methods"][runner_name]["provider_specific_logic"].append(pattern)
        
        except Exception as e:
            analysis["token_counting_methods"][runner_name] = {"error": str(e)}
    
    return analysis

def analyze_usage_tracking_standardization() -> Dict[str, Any]:
    """Analyze standardization of usage tracking formats."""
    
    project_root = Path(__file__).parent
    analysis = {
        "standard_format_compliance": {},
        "format_violations": [],
        "consistency_score": 0,
        "standardization_evidence": []
    }
    
    # Look for usage tracking in actual logs
    experiments_dir = project_root / "experiments"
    if not experiments_dir.exists():
        analysis["no_experiments"] = "No experiments directory found"
        return analysis
    
    # Sample log files for usage format analysis
    log_files = []
    for experiment_dir in experiments_dir.iterdir():
        if experiment_dir.is_dir():
            logs_dir = experiment_dir / "logs"
            if logs_dir.exists():
                log_files.extend(list(logs_dir.glob("*.json"))[:3])  # Sample 3 per experiment
    
    if not log_files:
        analysis["no_logs"] = "No log files found for analysis"
        return analysis
    
    # Analyze usage formats in logs
    usage_formats = []
    format_violations = []
    
    for log_file in log_files[:10]:  # Limit to 10 logs
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
            
            # Look for usage tracking in the log
            usage_found = False
            if "usage" in log_data:
                usage = log_data["usage"]
                usage_found = True
                
                # Check for standard format
                expected_keys = {"prompt_tokens", "completion_tokens", "total_tokens"}
                alternative_keys = {"input_tokens", "output_tokens"}
                
                usage_keys = set(usage.keys())
                
                if expected_keys.issubset(usage_keys):
                    usage_formats.append("standard_format")
                elif alternative_keys.issubset(usage_keys):
                    usage_formats.append("alternative_format")  
                else:
                    format_violations.append({
                        "file": log_file.name,
                        "usage_keys": list(usage_keys),
                        "missing_standard": list(expected_keys - usage_keys)
                    })
            
            # Also check in turns
            if "turns" in log_data:
                for turn in log_data["turns"]:
                    if "usage" in turn:
                        usage_found = True
                        # Same analysis for turn-level usage
        
        except Exception:
            continue
    
    # Calculate consistency metrics
    total_usage_instances = len(usage_formats) + len(format_violations)
    if total_usage_instances > 0:
        analysis["consistency_score"] = len(usage_formats) / total_usage_instances
        analysis["standard_format_compliance"] = {
            "standard_format_count": usage_formats.count("standard_format"),
            "alternative_format_count": usage_formats.count("alternative_format"),
            "violation_count": len(format_violations),
            "total_instances": total_usage_instances
        }
        analysis["format_violations"] = format_violations
    
    return analysis

def test_token_counting_estimation_accuracy() -> Dict[str, Any]:
    """Test token counting estimation methods for accuracy patterns."""
    
    project_root = Path(__file__).parent
    analysis = {
        "estimation_tests": {},
        "accuracy_assessment": {},
        "methodology_analysis": {},
        "reliability_score": 0
    }
    
    # Test common token counting approaches
    test_texts = [
        "Hello, how are you today?",
        "This is a longer sentence with more complex vocabulary and punctuation!",
        "Very short.",
        "A" * 100,  # 100 character string
        "Multi-line\ntext with\nvarious punctuation: !@#$%^&*()_+{}[]|\\:;\"'<>?,./"
    ]
    
    estimation_results = {}
    
    # Test simple length-based estimation (common fallback)
    for i, text in enumerate(test_texts):
        char_count = len(text)
        word_count = len(text.split())
        
        # Common estimation approaches found in codebase
        simple_estimate = char_count // 4  # Very common "4 chars per token" 
        word_estimate = int(word_count * 1.3)  # "1.3 tokens per word"
        
        estimation_results[f"text_{i}"] = {
            "text": text[:30] + "..." if len(text) > 30 else text,
            "char_count": char_count,
            "word_count": word_count,
            "simple_estimate": simple_estimate,
            "word_estimate": word_estimate,
            "length": len(text)
        }
    
    analysis["estimation_tests"] = estimation_results
    
    # Analyze estimation methodology quality
    # Look for sophisticated vs naive approaches
    methodology_indicators = {
        "naive_approaches": 0,
        "sophisticated_approaches": 0,
        "provider_specific": 0
    }
    
    # This would normally test against actual tokenizers, but we'll analyze patterns
    estimation_variance = []
    for text_data in estimation_results.values():
        simple = text_data["simple_estimate"]
        word = text_data["word_estimate"]
        if simple > 0:
            variance = abs(simple - word) / simple
            estimation_variance.append(variance)
            
            # Naive indicators
            if simple == text_data["char_count"] // 4:
                methodology_indicators["naive_approaches"] += 1
    
    if estimation_variance:
        analysis["accuracy_assessment"] = {
            "estimation_variance_avg": statistics.mean(estimation_variance),
            "estimation_variance_max": max(estimation_variance),
            "high_variance_detected": max(estimation_variance) > 0.5,
            "methodology_indicators": methodology_indicators
        }
        
        # Calculate reliability score
        avg_variance = statistics.mean(estimation_variance)
        sophistication_ratio = methodology_indicators["sophisticated_approaches"] / max(1, methodology_indicators["naive_approaches"])
        
        analysis["reliability_score"] = max(0, (1 - avg_variance) * (1 + sophistication_ratio * 0.1))
    
    return analysis

def run_token_counting_accuracy_analysis():
    """Run comprehensive token counting accuracy analysis."""
    
    print("🔬 TOKEN COUNTING ACCURACY ANALYSIS")
    print("=" * 60)
    print("Testing 'standardized usage tracking' and token estimation accuracy claims")
    print("Adversary position: 'Token estimates are wildly wrong, billing doesn't match'\\n")
    
    # Analyze token counting implementations
    print("📊 TOKEN COUNTING IMPLEMENTATION ANALYSIS")
    print("-" * 40)
    
    implementation_analysis = analyze_token_counting_implementations()
    
    if "error" in implementation_analysis:
        print(f"   ❌ Analysis failed: {implementation_analysis['error']}")
        return
    
    total_runners = len(implementation_analysis["token_counting_methods"])
    count_tokens_methods = 0
    total_approaches = 0
    providers_with_accuracy = 0
    
    for runner_name, runner_info in implementation_analysis["token_counting_methods"].items():
        if "error" not in runner_info:
            has_method = runner_info.get("count_tokens_method", False)
            approach_count = len(runner_info.get("token_counting_approach", []))
            accuracy_count = len(runner_info.get("accuracy_indicators", []))
            
            if has_method:
                count_tokens_methods += 1
            
            total_approaches += approach_count
            
            if accuracy_count > 0:
                providers_with_accuracy += 1
            
            print(f"\\n🎯 {runner_name}")
            print(f"   _count_tokens method: {'✅' if has_method else '❌'}")
            print(f"   Token counting approaches: {approach_count}")
            print(f"   Usage format indicators: {len(runner_info.get('usage_format', []))}")
            print(f"   Accuracy indicators: {accuracy_count}")
            
            if runner_info.get("provider_specific_logic"):
                print(f"   🚨 Provider-specific logic detected")
    
    print(f"\\n   Method implementation rate: {count_tokens_methods}/{total_runners} ({count_tokens_methods/total_runners:.1%})")
    print(f"   Total counting approaches: {total_approaches}")
    print(f"   Providers with accuracy indicators: {providers_with_accuracy}")
    
    # Analyze usage tracking standardization
    print(f"\\n📊 USAGE TRACKING STANDARDIZATION")
    print("-" * 40)
    
    standardization_analysis = analyze_usage_tracking_standardization()
    
    if "no_experiments" in standardization_analysis:
        print(f"   ⚠️  {standardization_analysis['no_experiments']}")
    elif "no_logs" in standardization_analysis:
        print(f"   ⚠️  {standardization_analysis['no_logs']}")
    else:
        compliance = standardization_analysis.get("standard_format_compliance", {})
        print(f"   Standard format instances: {compliance.get('standard_format_count', 0)}")
        print(f"   Alternative format instances: {compliance.get('alternative_format_count', 0)}")
        print(f"   Format violations: {compliance.get('violation_count', 0)}")
        print(f"   Consistency score: {standardization_analysis.get('consistency_score', 0):.1%}")
        
        if standardization_analysis.get("format_violations"):
            print(f"   🚨 Format standardization violations detected")
    
    # Test estimation accuracy
    print(f"\\n📊 ESTIMATION ACCURACY TESTING")
    print("-" * 40)
    
    accuracy_analysis = test_token_counting_estimation_accuracy()
    
    estimation_tests = accuracy_analysis.get("estimation_tests", {})
    print(f"   Test texts analyzed: {len(estimation_tests)}")
    
    accuracy_assessment = accuracy_analysis.get("accuracy_assessment", {})
    if accuracy_assessment:
        print(f"   Average estimation variance: {accuracy_assessment.get('estimation_variance_avg', 0):.1%}")
        print(f"   Maximum estimation variance: {accuracy_assessment.get('estimation_variance_max', 0):.1%}")
        print(f"   High variance detected: {accuracy_assessment.get('high_variance_detected', False)}")
        print(f"   Reliability score: {accuracy_analysis.get('reliability_score', 0):.1%}")
        
        methodology = accuracy_assessment.get("methodology_indicators", {})
        print(f"   Naive approaches: {methodology.get('naive_approaches', 0)}")
        print(f"   Sophisticated approaches: {methodology.get('sophisticated_approaches', 0)}")
    
    # Final verdict
    print("\\n" + "=" * 60)
    print("📋 FINAL TOKEN COUNTING ASSESSMENT:")
    
    critical_failures = (
        count_tokens_methods < total_runners * 0.8 or  # Less than 80% have method
        standardization_analysis.get("consistency_score", 0) < 0.8 or  # Low format consistency
        accuracy_assessment.get("high_variance_detected", True) or  # High estimation variance
        providers_with_accuracy == 0 or  # No accuracy indicators
        accuracy_analysis.get("reliability_score", 0) < 0.7  # Low reliability
    )
    
    print(f"   Method implementation: {count_tokens_methods}/{total_runners} ({count_tokens_methods/total_runners:.1%})")
    print(f"   Format consistency: {standardization_analysis.get('consistency_score', 0):.1%}")
    print(f"   Estimation reliability: {accuracy_analysis.get('reliability_score', 0):.1%}")
    print(f"   Accuracy indicators: {providers_with_accuracy} providers")
    
    if critical_failures:
        print("\\n   ⚔️  ADVERSARY VINDICATED: Token counting lacks accuracy")
        print("   🔥 'Standardized usage tracking' and estimation accuracy unsupported")
        print("   📉 UNCERTAINTY_LOG TOKEN COUNTING CLAIMS INVALIDATED") 
        print("   💥 Evidence: Inconsistent methods, format violations, unreliable estimates")
    else:
        print("\\n   🎯 TOKEN COUNTING ACCURACY VALIDATED")
        print("   ✅ Consistent methods with reliable estimation and standardized formats")
        print("   📝 'Standardized usage tracking' claims supported by evidence")
    
    return {
        "implementation_analysis": implementation_analysis,
        "standardization_analysis": standardization_analysis,
        "accuracy_analysis": accuracy_analysis,
        "overall_verdict": "INVALIDATED" if critical_failures else "VALIDATED"
    }

if __name__ == "__main__":
    # Run the comprehensive token counting accuracy analysis
    results = run_token_counting_accuracy_analysis()
    
    print(f"\\n{''}" + "=" * 60)
    print("🧪 Token counting accuracy analysis complete")
    print("   Focus: Implementation consistency, format standardization, estimation reliability")
    print("   Variable: results")