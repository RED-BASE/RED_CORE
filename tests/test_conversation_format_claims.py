#!/usr/bin/env python3
"""
Conversation format standardization test to verify "unified architecture" claims across providers.

The adversary's challenge: "Prove your conversation format handling is actually unified. 
I bet each provider uses completely different formats that break consistency, create attack 
surface variations, and make your 'standardized' approach a format chaos nightmare."

This test validates conversation format claims:
- Unified conversation handling across all 27+ models
- Consistent format conversion without information loss
- Attack surface uniformity across different providers
- Standardized conversation history management
- Provider-specific format handling quality

If format handling is unified, we should see:
- Consistent conversation structures across all providers
- No information loss during format conversion
- Uniform attack surface regardless of provider
- Standardized conversation history handling

If the adversary is right, we should see:
- Wildly different conversation formats creating inconsistencies
- Information loss or corruption during format conversion
- Provider-specific attack surface variations
- Format chaos disguised as "unified architecture"
"""

import ast
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from app.core.context import ConversationHistory, MessageTurn
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False

def analyze_conversation_format_implementations() -> Dict[str, Any]:
    """Analyze actual conversation format implementations across providers."""
    
    project_root = Path(__file__).parent
    api_runners_dir = project_root / "app" / "api_runners"
    
    analysis = {
        "format_implementations": {},
        "format_inconsistencies": [],
        "conversion_methods": {},
        "unified_architecture_violations": []
    }
    
    if not api_runners_dir.exists():
        analysis["error"] = "API runners directory not found"
        return analysis
    
    # Find all API runner files
    runner_files = list(api_runners_dir.glob("*_runner.py"))
    
    for runner_file in runner_files:
        runner_name = runner_file.stem
        
        analysis["format_implementations"][runner_name] = {
            "file_name": runner_name,
            "format_methods": [],
            "conversation_handling": [],
            "system_prompt_handling": [],
            "message_structure": [],
            "format_violations": []
        }
        
        try:
            with open(runner_file, 'r') as f:
                content = f.read()
            
            # Look for format conversion methods
            format_method_patterns = [
                r"to_.*_format",
                r"convert.*conversation", 
                r"format.*conversation",
                r"build.*messages",
                r"prepare.*conversation"
            ]
            
            for pattern in format_method_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["format_implementations"][runner_name]["format_methods"].extend(matches)
            
            # Look for conversation handling patterns
            conversation_patterns = [
                "conversation_history",
                "message_history", 
                "context",
                "turns",
                "chat_history",
                "dialog_history"
            ]
            
            for pattern in conversation_patterns:
                if pattern in content.lower():
                    analysis["format_implementations"][runner_name]["conversation_handling"].append(pattern)
            
            # Look for system prompt handling
            system_prompt_patterns = [
                "system_prompt",
                "system_message",
                "instructions",
                "system.*role",
                "system.*content"
            ]
            
            for pattern in system_prompt_patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    analysis["format_implementations"][runner_name]["system_prompt_handling"].append(pattern)
            
            # Look for message structure patterns
            message_structure_patterns = [
                r'\{"role":\s*".*?",\s*"content"',
                r"HUMAN_PROMPT",
                r"AI_PROMPT", 
                r"parts.*text",
                r"role.*user.*assistant",
                r"role.*model"
            ]
            
            for pattern in message_structure_patterns:
                import re
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis["format_implementations"][runner_name]["message_structure"].extend(matches)
            
            # Look for format-specific violations
            violation_indicators = [
                "# HACK",
                "# WORKAROUND", 
                "# FIXME",
                "# TODO.*format",
                "special.*case",
                "provider.*specific"
            ]
            
            for indicator in violation_indicators:
                import re
                if re.search(indicator, content, re.IGNORECASE):
                    analysis["format_implementations"][runner_name]["format_violations"].append(indicator)
        
        except Exception as e:
            analysis["format_implementations"][runner_name] = {"error": str(e)}
    
    return analysis

def analyze_format_conversion_consistency() -> Dict[str, Any]:
    """Analyze consistency of format conversion methods."""
    
    project_root = Path(__file__).parent
    context_file = project_root / "app" / "core" / "context.py"
    
    analysis = {
        "conversion_methods_found": [],
        "format_standardization": {},
        "information_loss_risks": [],
        "consistency_violations": []
    }
    
    if not context_file.exists():
        analysis["error"] = "Context file not found"
        return analysis
    
    try:
        with open(context_file, 'r') as f:
            content = f.read()
        
        # Parse the Python file to analyze format conversion methods
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "format" in node.name.lower():
                method_analysis = {
                    "method_name": node.name,
                    "complexity": len([n for n in ast.walk(node) if isinstance(n, ast.If)]),
                    "provider_specific_handling": [],
                    "information_preservation": []
                }
                
                # Look for provider-specific handling
                func_content = ast.get_source_segment(content, node) if hasattr(ast, 'get_source_segment') else ""
                if func_content:
                    provider_patterns = ["openai", "claude", "gemini", "anthropic", "google"]
                    for provider in provider_patterns:
                        if provider in func_content.lower():
                            method_analysis["provider_specific_handling"].append(provider)
                
                analysis["conversion_methods_found"].append(method_analysis)
        
        # Look for format standardization evidence
        standardization_indicators = [
            "standard.*format",
            "unified.*format", 
            "consistent.*format",
            "normalize.*format"
        ]
        
        for indicator in standardization_indicators:
            import re
            if re.search(indicator, content, re.IGNORECASE):
                analysis["format_standardization"][indicator] = True
        
        # Look for information loss risks
        loss_risk_patterns = [
            "truncate",
            "limit.*length",
            "max.*tokens",
            "strip.*system",
            "embed.*system",
            "concatenate"
        ]
        
        for pattern in loss_risk_patterns:
            import re
            if re.search(pattern, content, re.IGNORECASE):
                analysis["information_loss_risks"].append(pattern)
    
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def test_conversation_format_consistency() -> Dict[str, Any]:
    """Test actual conversation format consistency if context is available."""
    
    if not CONTEXT_AVAILABLE:
        return {"error": "ConversationHistory not available for testing"}
    
    analysis = {
        "format_test_results": {},
        "consistency_check": {},
        "information_preservation": {},
        "format_chaos_detected": False
    }
    
    try:
        # Create a test conversation
        conversation = ConversationHistory()
        conversation.system_prompt = "You are a helpful assistant."
        
        # Add some test turns
        conversation.append_user("Hello, how are you?")
        conversation.append_assistant("I'm doing well, thank you for asking!")
        conversation.append_user("Can you help me with a task?")
        conversation.append_assistant("Of course! I'd be happy to help. What do you need assistance with?")
        
        # Test format conversion methods
        format_methods = [
            ("to_openai_format", "OpenAI"),
            ("to_claude_format", "Claude"),
            ("to_gemini_format", "Gemini"),
            ("to_anthropic_format", "Anthropic")
        ]
        
        for method_name, provider in format_methods:
            if hasattr(conversation, method_name):
                try:
                    formatted = getattr(conversation, method_name)()
                    
                    analysis["format_test_results"][provider] = {
                        "method_exists": True,
                        "format_type": type(formatted).__name__,
                        "content_structure": str(type(formatted)),
                        "turn_count_preserved": len(conversation.turns),
                        "system_prompt_handled": bool(conversation.system_prompt)
                    }
                    
                    # Check for information preservation
                    if isinstance(formatted, (list, dict, str)):
                        content_str = str(formatted)
                        user_content_preserved = "Hello, how are you?" in content_str
                        assistant_content_preserved = "I'm doing well" in content_str
                        system_prompt_preserved = "helpful assistant" in content_str
                        
                        analysis["format_test_results"][provider]["information_preservation"] = {
                            "user_content": user_content_preserved,
                            "assistant_content": assistant_content_preserved,
                            "system_prompt": system_prompt_preserved,
                            "preservation_score": sum([user_content_preserved, assistant_content_preserved, system_prompt_preserved]) / 3
                        }
                
                except Exception as e:
                    analysis["format_test_results"][provider] = {
                        "method_exists": True,
                        "error": str(e)
                    }
            else:
                analysis["format_test_results"][provider] = {
                    "method_exists": False
                }
        
        # Check for format consistency
        successful_formats = [
            provider for provider, result in analysis["format_test_results"].items()
            if result.get("method_exists") and "error" not in result
        ]
        
        analysis["consistency_check"] = {
            "total_providers": len(format_methods),
            "successful_formats": len(successful_formats),
            "format_success_rate": len(successful_formats) / len(format_methods),
            "consistent_formats": len(successful_formats) == len(format_methods)
        }
        
        # Detect format chaos
        preservation_scores = []
        for provider, result in analysis["format_test_results"].items():
            if "information_preservation" in result:
                preservation_scores.append(result["information_preservation"]["preservation_score"])
        
        if preservation_scores:
            avg_preservation = sum(preservation_scores) / len(preservation_scores)
            analysis["information_preservation"] = {
                "average_preservation_score": avg_preservation,
                "high_preservation": avg_preservation > 0.9,
                "preservation_scores": preservation_scores
            }
            
            # Format chaos detected if low preservation or high variance
            preservation_variance = sum((score - avg_preservation) ** 2 for score in preservation_scores) / len(preservation_scores) if len(preservation_scores) > 1 else 0
            analysis["format_chaos_detected"] = avg_preservation < 0.8 or preservation_variance > 0.1
    
    except Exception as e:
        analysis["error"] = str(e)
    
    return analysis

def run_conversation_format_analysis():
    """Run comprehensive conversation format analysis."""
    
    print("🔬 CONVERSATION FORMAT CONSISTENCY ANALYSIS")
    print("=" * 60)
    print("Testing 'unified architecture' and format standardization claims")
    print("Adversary position: 'Format handling is chaos disguised as unified architecture'\\n")
    
    # Analyze format implementations
    print("📊 FORMAT IMPLEMENTATION ANALYSIS")
    print("-" * 40)
    
    implementation_analysis = analyze_conversation_format_implementations()
    
    if "error" in implementation_analysis:
        print(f"   ❌ Analysis failed: {implementation_analysis['error']}")
        return
    
    total_runners = len(implementation_analysis["format_implementations"])
    print(f"   API runners analyzed: {total_runners}")
    
    # Count format methods and violations
    total_format_methods = 0
    total_violations = 0
    providers_with_methods = 0
    
    for runner_name, runner_info in implementation_analysis["format_implementations"].items():
        if "error" not in runner_info:
            method_count = len(runner_info.get("format_methods", []))
            violation_count = len(runner_info.get("format_violations", []))
            
            total_format_methods += method_count
            total_violations += violation_count
            
            if method_count > 0:
                providers_with_methods += 1
            
            print(f"\\n🎯 {runner_name}")
            print(f"   Format methods: {method_count}")
            print(f"   Conversation handling: {len(runner_info.get('conversation_handling', []))}")
            print(f"   System prompt handling: {len(runner_info.get('system_prompt_handling', []))}")
            print(f"   Format violations: {violation_count}")
            
            if violation_count > 0:
                print(f"   🚨 Format violations detected in {runner_name}")
    
    # Analyze format conversion consistency
    print(f"\\n📊 FORMAT CONVERSION CONSISTENCY")
    print("-" * 40)
    
    conversion_analysis = analyze_format_conversion_consistency()
    
    if "error" in conversion_analysis:
        print(f"   ⚠️  Conversion analysis: {conversion_analysis['error']}")
    else:
        print(f"   Conversion methods found: {len(conversion_analysis['conversion_methods_found'])}")
        print(f"   Standardization indicators: {len(conversion_analysis['format_standardization'])}")
        print(f"   Information loss risks: {len(conversion_analysis['information_loss_risks'])}")
        
        if conversion_analysis['information_loss_risks']:
            print(f"   🚨 Information loss risks detected:")
            for risk in conversion_analysis['information_loss_risks'][:3]:
                print(f"      • {risk}")
    
    # Test format consistency if possible
    print(f"\\n📊 FORMAT CONSISTENCY TESTING")
    print("-" * 40)
    
    consistency_test = test_conversation_format_consistency()
    
    if "error" in consistency_test:
        print(f"   ⚠️  Consistency testing: {consistency_test['error']}")
    else:
        consistency_check = consistency_test.get("consistency_check", {})
        print(f"   Format success rate: {consistency_check.get('format_success_rate', 0):.1%}")
        print(f"   Successful formats: {consistency_check.get('successful_formats', 0)}/{consistency_check.get('total_providers', 0)}")
        print(f"   Format chaos detected: {consistency_test.get('format_chaos_detected', False)}")
        
        preservation = consistency_test.get("information_preservation", {})
        if preservation:
            print(f"   Average preservation score: {preservation.get('average_preservation_score', 0):.1%}")
            print(f"   High preservation: {preservation.get('high_preservation', False)}")
    
    # Final verdict
    print("\\n" + "=" * 60)
    print("📋 FINAL CONVERSATION FORMAT ASSESSMENT:")
    
    critical_failures = (
        total_violations > 5 or
        providers_with_methods < total_runners * 0.8 or
        len(conversion_analysis.get('information_loss_risks', [])) > 3 or
        consistency_test.get('format_chaos_detected', False) or
        consistency_test.get('consistency_check', {}).get('format_success_rate', 0) < 0.8
    )
    
    print(f"   Format method coverage: {providers_with_methods}/{total_runners}")
    print(f"   Total format violations: {total_violations}")
    print(f"   Information loss risks: {len(conversion_analysis.get('information_loss_risks', []))}")
    print(f"   Format consistency rate: {consistency_test.get('consistency_check', {}).get('format_success_rate', 0):.1%}")
    
    if critical_failures:
        print("\\n   ⚔️  ADVERSARY VINDICATED: Format handling lacks unification")
        print("   🔥 'Unified architecture' is format chaos with provider-specific hacks")
        print("   📉 UNCERTAINTY_LOG FORMAT CLAIMS INVALIDATED")
        print("   💥 Evidence: Format violations, inconsistent conversion, information loss")
    else:
        print("\\n   🎯 CONVERSATION FORMAT UNIFICATION VALIDATED")
        print("   ✅ Consistent format handling with minimal information loss")
        print("   📝 'Unified architecture' claims supported by evidence")
    
    return {
        "implementation_analysis": implementation_analysis,
        "conversion_analysis": conversion_analysis,
        "consistency_test": consistency_test,
        "overall_verdict": "INVALIDATED" if critical_failures else "VALIDATED"
    }

if __name__ == "__main__":
    # Run the comprehensive conversation format analysis
    results = run_conversation_format_analysis()
    
    print(f"\\n{''}" + "=" * 60)
    print("🧪 Conversation format analysis complete")
    print("   Focus: Format consistency, information preservation, unified architecture")
    print("   Variable: results")