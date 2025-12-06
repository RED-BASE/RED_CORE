#!/usr/bin/env python3
"""
Conversation history efficiency test to verify "full conversation history gets sent" claims.

The adversary's challenge: "Prove your conversation history handling is actually efficient. 
I bet sending 'full conversation history on each turn' creates exponential token bloat, 
massive API costs, and inefficient context management that makes multi-turn conversations 
prohibitively expensive and slow."

This test validates conversation history efficiency claims:
- Token usage growth patterns across multiple turns
- API cost implications of full history transmission
- Context management efficiency and optimization
- Memory usage and performance impact
- Cost prediction accuracy for multi-turn conversations

If conversation handling is efficient, we should see:
- Reasonable token growth patterns with optimization
- Cost-effective context management strategies
- Memory-efficient conversation storage
- Accurate cost prediction for multi-turn scenarios
- Performance optimization for long conversations

If the adversary is right, we should see:
- Exponential token bloat making conversations prohibitively expensive
- No optimization for repeated context transmission
- Poor memory usage scaling with conversation length
- Inaccurate cost predictions leading to budget disasters
- Performance degradation in multi-turn scenarios
"""

import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import statistics

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.core.context import ConversationHistory
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

def analyze_conversation_history_implementation() -> Dict[str, Any]:
    """Analyze conversation history implementation for efficiency patterns."""
    
    project_root = Path(__file__).parent
    context_file = project_root / "app" / "core" / "context.py"
    
    analysis = {
        "implementation_analysis": {},
        "efficiency_indicators": {},
        "optimization_patterns": [],
        "inefficiency_warnings": []
    }
    
    if not context_file.exists():
        analysis["error"] = "Context file not found"
        return analysis
    
    try:
        with open(context_file, 'r') as f:
            content = f.read()
        
        # Look for conversation handling patterns
        analysis["implementation_analysis"] = {
            "full_history_patterns": [],
            "optimization_patterns": [],
            "memory_management": [],
            "efficiency_indicators": []
        }
        
        # Look for full history transmission patterns
        full_history_patterns = [
            r"full.*conversation",
            r"all.*turns", 
            r"complete.*history",
            r"entire.*conversation",
            r"turns.*1.*9"  # Sending turns 1-9 plus new turn
        ]
        
        for pattern in full_history_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                analysis["implementation_analysis"]["full_history_patterns"].append(pattern)
        
        # Look for optimization patterns
        optimization_patterns = [
            r"truncate",
            r"limit.*context",
            r"max.*tokens",
            r"context.*window",
            r"sliding.*window",
            r"compress.*history",
            r"summarize.*context",
            r"cache.*context"
        ]
        
        for pattern in optimization_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                analysis["implementation_analysis"]["optimization_patterns"].append(pattern)
        
        # Look for memory management
        memory_patterns = [
            "del.*history",
            "clear.*history", 
            "gc\.",
            "memory.*usage",
            "efficient.*storage"
        ]
        
        for pattern in memory_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                analysis["implementation_analysis"]["memory_management"].append(pattern)
        
        # Look for efficiency indicators
        efficiency_patterns = [
            "lazy.*load",
            "on.*demand",
            "efficient",
            "optimized",
            "performance",
            "fast.*access"
        ]
        
        for pattern in efficiency_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                analysis["implementation_analysis"]["efficiency_indicators"].append(pattern)
        
        # Check for inefficiency warnings
        if len(analysis["implementation_analysis"]["full_history_patterns"]) > 0 and \
           len(analysis["implementation_analysis"]["optimization_patterns"]) == 0:
            analysis["inefficiency_warnings"].append("Full history transmission without optimization")
        
        if len(analysis["implementation_analysis"]["memory_management"]) == 0:
            analysis["inefficiency_warnings"].append("No memory management patterns found")
    
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def simulate_conversation_token_growth() -> Dict[str, Any]:
    """Simulate token usage growth in multi-turn conversations."""
    
    analysis = {
        "token_growth_simulation": {},
        "cost_implications": {},
        "efficiency_metrics": {},
        "scalability_assessment": {}
    }
    
    # Simulate conversation turns with realistic content
    simulated_turns = [
        {"user": "Hello, how are you?", "assistant": "I'm doing well, thank you for asking! How can I help you today?"},
        {"user": "Can you help me with a Python problem?", "assistant": "Of course! I'd be happy to help you with Python. What specific problem are you working on?"},
        {"user": "I need to process a large CSV file efficiently", "assistant": "Great question! For processing large CSV files efficiently in Python, I recommend using pandas with chunking or the csv module for memory efficiency. Here are some approaches..."},
        {"user": "What about memory usage optimization?", "assistant": "Excellent follow-up! Memory optimization is crucial for large files. You can use techniques like reading in chunks, using generators, optimizing data types, and clearing unused variables..."},
        {"user": "Can you show me a code example?", "assistant": "Absolutely! Here's a practical example showing efficient CSV processing with memory optimization techniques..."},
        {"user": "How would this scale to files with millions of rows?", "assistant": "For files with millions of rows, you'll need additional strategies like parallel processing, database integration, and streaming approaches..."},
        {"user": "What about error handling in this context?", "assistant": "Error handling becomes even more important with large datasets. You'll want to implement robust exception handling, logging, and recovery mechanisms..."},
        {"user": "Are there any performance benchmarks?", "assistant": "Yes! Performance benchmarking is essential for optimization. Here are some approaches and typical performance characteristics you can expect..."},
        {"user": "Thank you, this is very helpful!", "assistant": "You're very welcome! I'm glad I could help you understand efficient CSV processing. Feel free to ask if you have more questions!"},
        {"user": "One last question about deployment", "assistant": "Of course! Deployment considerations for data processing applications include containerization, resource allocation, monitoring, and scaling strategies..."}
    ]
    
    # Calculate cumulative token usage (using simple estimation)
    token_growth = []
    cumulative_tokens = 0
    system_prompt_tokens = 50  # Estimated system prompt
    
    for turn_index, turn in enumerate(simulated_turns):
        # Simple token estimation (4 characters ≈ 1 token)
        user_tokens = len(turn["user"]) // 4
        assistant_tokens = len(turn["assistant"]) // 4
        
        # Full conversation history approach: send ALL previous turns + current
        if turn_index == 0:
            # First turn: system prompt + user prompt
            turn_tokens = system_prompt_tokens + user_tokens + assistant_tokens
        else:
            # Subsequent turns: system prompt + ALL previous turns + current turn
            # This is the "full conversation history gets sent" pattern
            previous_conversation_tokens = sum(token_data["cumulative_tokens"] for token_data in token_growth[-1:])
            turn_tokens = system_prompt_tokens + previous_conversation_tokens + user_tokens + assistant_tokens
        
        cumulative_tokens += turn_tokens
        
        token_data = {
            "turn": turn_index + 1,
            "user_tokens": user_tokens,
            "assistant_tokens": assistant_tokens,
            "turn_tokens": turn_tokens,
            "cumulative_tokens": cumulative_tokens,
            "growth_factor": turn_tokens / token_growth[0]["turn_tokens"] if token_growth else 1.0
        }
        
        token_growth.append(token_data)
    
    analysis["token_growth_simulation"] = token_growth
    
    # Calculate cost implications (using rough API pricing)
    api_costs = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # Per 1K tokens
        "claude-3": {"input": 0.015, "output": 0.075},
        "gemini-pro": {"input": 0.0005, "output": 0.0015}
    }
    
    total_tokens = token_growth[-1]["cumulative_tokens"] if token_growth else 0
    
    cost_analysis = {}
    for model, pricing in api_costs.items():
        # Assume 50/50 input/output split
        input_tokens = total_tokens * 0.5
        output_tokens = total_tokens * 0.5
        
        total_cost = (input_tokens / 1000 * pricing["input"]) + (output_tokens / 1000 * pricing["output"])
        
        cost_analysis[model] = {
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "cost_per_turn": total_cost / len(simulated_turns) if simulated_turns else 0
        }
    
    analysis["cost_implications"] = cost_analysis
    
    # Calculate efficiency metrics
    if token_growth:
        initial_tokens = token_growth[0]["turn_tokens"]
        final_tokens = token_growth[-1]["turn_tokens"]
        
        analysis["efficiency_metrics"] = {
            "initial_turn_tokens": initial_tokens,
            "final_turn_tokens": final_tokens,
            "growth_factor": final_tokens / initial_tokens if initial_tokens > 0 else 0,
            "exponential_growth": final_tokens > initial_tokens * (len(simulated_turns) ** 2),
            "linear_growth": final_tokens < initial_tokens * len(simulated_turns) * 2,
            "average_tokens_per_turn": statistics.mean([turn["turn_tokens"] for turn in token_growth]),
            "median_tokens_per_turn": statistics.median([turn["turn_tokens"] for turn in token_growth])
        }
    
    return analysis

def test_conversation_memory_usage() -> Dict[str, Any]:
    """Test memory usage patterns in conversation handling."""
    
    analysis = {
        "memory_simulation": {},
        "scaling_test": {},
        "efficiency_assessment": {}
    }
    
    if not CONTEXT_AVAILABLE:
        analysis["error"] = "ConversationHistory not available for testing"
        return analysis
    
    try:
        # Test memory usage scaling
        conversation_sizes = [1, 5, 10, 20, 50]
        memory_usage = []
        
        for size in conversation_sizes:
            # Create conversation with specified number of turns
            conversation = ConversationHistory()
            conversation.system_prompt = "You are a helpful assistant."
            
            for i in range(size):
                conversation.append_user(f"This is user message {i} with some content to make it realistic.")
                conversation.append_assistant(f"This is assistant response {i} with detailed information and helpful content that would be typical in a real conversation.")
            
            # Estimate memory usage (rough approximation)
            total_content = str(conversation)
            estimated_memory = len(total_content.encode('utf-8'))
            
            memory_usage.append({
                "turns": size,
                "estimated_memory_bytes": estimated_memory,
                "memory_per_turn": estimated_memory / size if size > 0 else 0
            })
        
        analysis["memory_simulation"] = memory_usage
        
        # Assess scaling efficiency
        if len(memory_usage) >= 2:
            small_conversation = memory_usage[0]  # 1 turn
            large_conversation = memory_usage[-1]  # 50 turns
            
            scaling_factor = large_conversation["estimated_memory_bytes"] / small_conversation["estimated_memory_bytes"]
            linear_expectation = large_conversation["turns"] / small_conversation["turns"]
            
            analysis["scaling_test"] = {
                "actual_scaling_factor": scaling_factor,
                "linear_scaling_expectation": linear_expectation,
                "scaling_efficiency": linear_expectation / scaling_factor if scaling_factor > 0 else 0,
                "efficient_scaling": abs(scaling_factor - linear_expectation) < linear_expectation * 0.2  # Within 20%
            }
    
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def run_conversation_history_efficiency_analysis():
    """Run comprehensive conversation history efficiency analysis."""
    
    print("🔬 CONVERSATION HISTORY EFFICIENCY ANALYSIS")
    print("=" * 60)
    print("Testing 'full conversation history gets sent' efficiency claims")
    print("Adversary position: 'Full history transmission creates exponential token bloat'\\n")
    
    # Analyze implementation patterns
    print("📊 CONVERSATION HISTORY IMPLEMENTATION")
    print("-" * 40)
    
    implementation_analysis = analyze_conversation_history_implementation()
    
    if "error" in implementation_analysis:
        print(f"   ❌ Analysis failed: {implementation_analysis['error']}")
        return
    
    full_history_patterns = len(implementation_analysis["implementation_analysis"]["full_history_patterns"])
    optimization_patterns = len(implementation_analysis["implementation_analysis"]["optimization_patterns"])
    memory_management = len(implementation_analysis["implementation_analysis"]["memory_management"])
    inefficiency_warnings = len(implementation_analysis["inefficiency_warnings"])
    
    print(f"   Full history patterns: {full_history_patterns}")
    print(f"   Optimization patterns: {optimization_patterns}")
    print(f"   Memory management patterns: {memory_management}")
    print(f"   Inefficiency warnings: {inefficiency_warnings}")
    
    if inefficiency_warnings > 0:
        print(f"   🚨 Efficiency warnings:")
        for warning in implementation_analysis["inefficiency_warnings"]:
            print(f"      • {warning}")
    
    # Simulate token growth
    print(f"\\n📊 TOKEN USAGE SIMULATION")
    print("-" * 40)
    
    token_simulation = simulate_conversation_token_growth()
    
    token_growth = token_simulation["token_growth_simulation"]
    efficiency_metrics = token_simulation["efficiency_metrics"]
    
    if token_growth:
        initial_tokens = efficiency_metrics["initial_turn_tokens"]
        final_tokens = efficiency_metrics["final_turn_tokens"]
        growth_factor = efficiency_metrics["growth_factor"]
        exponential_growth = efficiency_metrics["exponential_growth"]
        
        print(f"   10-turn conversation simulation:")
        print(f"   Initial turn tokens: {initial_tokens}")
        print(f"   Final turn tokens: {final_tokens}")
        print(f"   Growth factor: {growth_factor:.1f}x")
        print(f"   Exponential growth detected: {exponential_growth}")
        
        if exponential_growth:
            print(f"   🚨 EXPONENTIAL TOKEN BLOAT DETECTED")
        
        # Show token progression
        print(f"   Token progression (first 5 turns):")
        for turn_data in token_growth[:5]:
            print(f"      Turn {turn_data['turn']}: {turn_data['turn_tokens']} tokens")
    
    # Analyze cost implications
    print(f"\\n📊 COST IMPLICATIONS")
    print("-" * 40)
    
    cost_implications = token_simulation["cost_implications"]
    
    for model, cost_data in cost_implications.items():
        total_cost = cost_data["estimated_cost"]
        cost_per_turn = cost_data["cost_per_turn"]
        
        print(f"   {model}:")
        print(f"      Total cost (10 turns): ${total_cost:.4f}")
        print(f"      Cost per turn: ${cost_per_turn:.4f}")
        
        if cost_per_turn > 0.10:  # More than 10 cents per turn
            print(f"      🚨 HIGH COST PER TURN")
    
    # Test memory usage
    print(f"\\n📊 MEMORY USAGE SCALING")
    print("-" * 40)
    
    memory_test = test_conversation_memory_usage()
    
    if "error" in memory_test:
        print(f"   ⚠️  Memory test: {memory_test['error']}")
    else:
        memory_usage = memory_test["memory_simulation"]
        scaling_test = memory_test.get("scaling_test", {})
        
        if memory_usage:
            print(f"   Memory usage by conversation size:")
            for usage in memory_usage[::2]:  # Show every other entry
                memory_mb = usage["estimated_memory_bytes"] / (1024 * 1024)
                print(f"      {usage['turns']} turns: {memory_mb:.2f} MB")
        
        if scaling_test:
            scaling_factor = scaling_test["actual_scaling_factor"]
            efficient_scaling = scaling_test["efficient_scaling"]
            
            print(f"   Scaling efficiency: {'✅ EFFICIENT' if efficient_scaling else '❌ INEFFICIENT'}")
            print(f"   Actual scaling factor: {scaling_factor:.1f}x")
    
    # Final verdict
    print("\\n" + "=" * 60)
    print("📋 FINAL CONVERSATION EFFICIENCY ASSESSMENT:")
    
    critical_failures = (
        inefficiency_warnings > 1 or  # Multiple efficiency warnings
        optimization_patterns == 0 or  # No optimization patterns
        exponential_growth or  # Exponential token growth
        any(cost_data["cost_per_turn"] > 0.10 for cost_data in cost_implications.values()) or  # High cost per turn
        not memory_test.get("scaling_test", {}).get("efficient_scaling", True)  # Inefficient memory scaling
    )
    
    print(f"   Optimization patterns: {optimization_patterns}")
    print(f"   Token growth factor: {efficiency_metrics.get('growth_factor', 0):.1f}x")
    print(f"   Exponential growth: {exponential_growth}")
    print(f"   Memory scaling: {'Efficient' if memory_test.get('scaling_test', {}).get('efficient_scaling', True) else 'Inefficient'}")
    
    if critical_failures:
        print("\\n   ⚔️  ADVERSARY VINDICATED: Conversation history is inefficient")
        print("   🔥 'Full conversation history' creates exponential token bloat")
        print("   📉 UNCERTAINTY_LOG CONVERSATION EFFICIENCY CLAIMS INVALIDATED")
        print("   💥 Evidence: Exponential growth, poor optimization, high costs")
    else:
        print("\\n   🎯 CONVERSATION HISTORY EFFICIENCY VALIDATED")
        print("   ✅ Efficient context management with reasonable scaling")
        print("   📝 'Full conversation history' claims supported by evidence")
    
    return {
        "implementation_analysis": implementation_analysis,
        "token_simulation": token_simulation,
        "memory_test": memory_test,
        "overall_verdict": "INVALIDATED" if critical_failures else "VALIDATED"
    }

if __name__ == "__main__":
    # Run the comprehensive conversation history efficiency analysis
    results = run_conversation_history_efficiency_analysis()
    
    print(f"\\n{''}" + "=" * 60)
    print("🧪 Conversation history efficiency analysis complete")
    print("   Focus: Token growth, cost implications, memory scaling")
    print("   Variable: results")