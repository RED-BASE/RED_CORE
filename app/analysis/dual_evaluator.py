"""
Dual Evaluation System for RED_CORE
Combines rule-based automated scoring with LLM-based evaluation.

This provides comprehensive assessment using both pattern-matching
and nuanced language model evaluation.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.core.log_schema import SessionLog, Turn
from app.core.logger import get_analysis_logger
from app.analysis.automated_scorer import AutomatedScorer
from app.analysis.llm_evaluator import LLMEvaluator

logger = get_analysis_logger()


class DualEvaluator:
    """Combines automated scoring with LLM evaluation."""
    
    def __init__(self, evaluator_model: str = "gemini-2.0-flash-lite"):
        """Initialize both evaluation systems."""
        self.automated_scorer = AutomatedScorer()
        self.llm_evaluator = LLMEvaluator(evaluator_model)
        self.evaluator_model = evaluator_model
        
        logger.info(f"Initialized dual evaluator with LLM model: {evaluator_model}")
    
    async def evaluate_session_complete(self, session_log: SessionLog) -> Dict[str, Any]:
        """
        Perform complete evaluation using both systems.
        Returns combined results and comparison metrics.
        """
        logger.info(f"Starting dual evaluation for session {session_log.isbn_run_id}")
        
        # Get automated scoring
        automated_results = self.automated_scorer.score_session(session_log)
        
        # Get LLM evaluation (modifies session_log in place)
        await self.llm_evaluator.evaluate_session(session_log, "comprehensive")
        
        # Extract LLM scores for comparison
        llm_scores = self._extract_llm_scores(session_log)
        
        # Compare scoring methods
        comparison = self._compare_scoring_methods(automated_results, llm_scores)
        
        return {
            "session_id": session_log.isbn_run_id,
            "model": session_log.model,
            "experiment": session_log.experiment_id,
            "automated_scoring": automated_results,
            "llm_evaluation": llm_scores,
            "scoring_comparison": comparison,
            "evaluation_metadata": {
                "llm_evaluator_model": self.evaluator_model,
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_turns": len(session_log.turns)
            }
        }
    
    def _extract_llm_scores(self, session_log: SessionLog) -> Dict[str, Any]:
        """Extract LLM evaluation scores from session log."""
        turn_evaluations = []
        refusal_scores = []
        drift_scores = []
        confidences = []
        
        for turn in session_log.turns:
            if turn.llm_evaluation:
                eval_data = {
                    "turn_index": turn.turn_index,
                    "refusal_score": turn.llm_evaluation.refusal_score,
                    "drift_score": turn.llm_evaluation.drift_score,
                    "confidence": turn.llm_evaluation.confidence,
                    "reasoning": turn.llm_evaluation.reasoning
                }
                turn_evaluations.append(eval_data)
                
                if turn.llm_evaluation.refusal_score:
                    refusal_scores.append(turn.llm_evaluation.refusal_score)
                if turn.llm_evaluation.drift_score is not None:
                    drift_scores.append(turn.llm_evaluation.drift_score)
                if turn.llm_evaluation.confidence:
                    confidences.append(turn.llm_evaluation.confidence)
        
        # Calculate aggregate metrics
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else None
        
        refusal_distribution = {}
        for score in refusal_scores:
            refusal_distribution[score] = refusal_distribution.get(score, 0) + 1
        
        return {
            "turn_evaluations": turn_evaluations,
            "aggregate_metrics": {
                "average_confidence": avg_confidence,
                "average_drift_score": avg_drift,
                "refusal_score_distribution": refusal_distribution,
                "total_evaluated_turns": len(turn_evaluations)
            }
        }
    
    def _compare_scoring_methods(self, automated: Dict[str, Any], llm: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results between automated and LLM scoring."""
        comparison = {
            "method_agreement": {},
            "score_correlations": {},
            "confidence_analysis": {}
        }
        
        # Compare refusal scoring if both available
        auto_refusal_breakdown = automated.get("refusal_breakdown", {})
        llm_refusal_dist = llm.get("aggregate_metrics", {}).get("refusal_score_distribution", {})
        
        if auto_refusal_breakdown and llm_refusal_dist:
            # Calculate agreement rate
            total_turns = automated.get("total_turns", 0)
            agreement_count = 0
            
            for score in [1, 2, 3]:
                auto_count = auto_refusal_breakdown.get(score, 0)
                llm_count = llm_refusal_dist.get(score, 0)
                agreement_count += min(auto_count, llm_count)
            
            agreement_rate = agreement_count / total_turns if total_turns > 0 else 0.0
            comparison["method_agreement"]["refusal_scoring"] = agreement_rate
        
        # Safety rate comparison
        auto_safety_rate = automated.get("safety_rate", 0.0)
        llm_avg_drift = llm.get("aggregate_metrics", {}).get("average_drift_score")
        
        if llm_avg_drift is not None:
            # Convert drift score to safety rate (inverse relationship)
            llm_safety_rate = 1.0 - llm_avg_drift
            comparison["score_correlations"]["safety_rate_difference"] = abs(auto_safety_rate - llm_safety_rate)
        
        # Confidence analysis
        llm_confidence = llm.get("aggregate_metrics", {}).get("average_confidence", 0.0)
        comparison["confidence_analysis"]["llm_average_confidence"] = llm_confidence
        
        return comparison
    
    async def evaluate_log_file(self, log_path: Path, output_path: Optional[Path] = None, update_in_place: bool = True) -> Optional[Dict[str, Any]]:
        """Evaluate a single log file with dual evaluation."""
        try:
            # Load log file
            with open(log_path, 'r') as f:
                data = json.load(f)
            
            session_log = SessionLog(**data)
            
            # Perform dual evaluation (modifies session_log in place)
            results = await self.evaluate_session_complete(session_log)
            
            if update_in_place:
                # Update the original log file with embedded scoring
                updated_data = session_log.model_dump()
                with open(log_path, 'w') as f:
                    json.dump(updated_data, f, indent=2)
                logger.info(f"Updated original log file with dual evaluation: {log_path}")
            
            # Also save separate analysis results if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Saved dual evaluation analysis to {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to dual evaluate log file {log_path}: {e}")
            return None
    
    async def batch_evaluate_directory(self, log_dir: Path, output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Evaluate all log files in a directory with dual evaluation."""
        log_files = [f for f in log_dir.glob("*.json") if f.name != "run_failures.txt" and not f.name.endswith("_dual_evaluated.json")]
        
        if not log_files:
            logger.warning(f"No JSON log files found in {log_dir}")
            return []
        
        # Default output directory: create 'dual_evaluated' subdirectory in same experiment folder
        if not output_dir:
            if "experiments" in str(log_dir):
                # Extract experiment folder (e.g., experiments/80k_hours_demo/logs -> experiments/80k_hours_demo/dual_evaluated)
                experiment_dir = log_dir.parent
                output_dir = experiment_dir / "dual_evaluated"
            else:
                # Fallback: create dual_evaluated next to logs directory
                output_dir = log_dir.parent / "dual_evaluated"
        
        output_dir.mkdir(exist_ok=True)
        logger.info(f"Starting dual evaluation of {len(log_files)} log files")
        logger.info(f"Output directory: {output_dir}")
        results = []
        
        for i, log_file in enumerate(log_files, 1):
            logger.info(f"Processing file {i}/{len(log_files)}: {log_file.name}")
            
            output_path = output_dir / f"{log_file.stem}_dual_evaluated.json"
            
            result = await self.evaluate_log_file(log_file, output_path, update_in_place=True)
            if result:
                results.append(result)
                
                # Log progress summary
                comparison = result.get("scoring_comparison", {})
                agreement = comparison.get("method_agreement", {}).get("refusal_scoring", "N/A")
                logger.info(f"  - Refusal scoring agreement: {agreement}")
        
        logger.info(f"Completed dual evaluation of {len(results)} files")
        
        # Generate summary report
        self._generate_summary_report(results, output_dir)
        
        return results
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], output_dir: Optional[Path]):
        """Generate a summary report of dual evaluation results."""
        if not results or not output_dir:
            return
        
        summary = {
            "evaluation_summary": {
                "total_sessions": len(results),
                "evaluator_model": self.evaluator_model,
                "timestamp": datetime.now().isoformat()
            },
            "agreement_analysis": {},
            "performance_metrics": {}
        }
        
        # Aggregate agreement rates
        refusal_agreements = []
        safety_rate_diffs = []
        avg_confidences = []
        
        for result in results:
            comparison = result.get("scoring_comparison", {})
            
            refusal_agreement = comparison.get("method_agreement", {}).get("refusal_scoring")
            if refusal_agreement is not None:
                refusal_agreements.append(refusal_agreement)
            
            safety_diff = comparison.get("score_correlations", {}).get("safety_rate_difference")
            if safety_diff is not None:
                safety_rate_diffs.append(safety_diff)
            
            confidence = result.get("llm_evaluation", {}).get("aggregate_metrics", {}).get("average_confidence")
            if confidence:
                avg_confidences.append(confidence)
        
        if refusal_agreements:
            summary["agreement_analysis"]["average_refusal_agreement"] = sum(refusal_agreements) / len(refusal_agreements)
            summary["agreement_analysis"]["refusal_agreement_range"] = [min(refusal_agreements), max(refusal_agreements)]
        
        if safety_rate_diffs:
            summary["agreement_analysis"]["average_safety_rate_difference"] = sum(safety_rate_diffs) / len(safety_rate_diffs)
        
        if avg_confidences:
            summary["performance_metrics"]["average_llm_confidence"] = sum(avg_confidences) / len(avg_confidences)
        
        # Save summary
        summary_path = output_dir / "dual_evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Generated summary report: {summary_path}")


# CLI interface
async def main():
    """CLI interface for dual evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Evaluation System for RED_CORE")
    parser.add_argument("--log-dir", help="Directory containing log files")
    parser.add_argument("--log-file", help="Single log file to evaluate")
    parser.add_argument("--output-dir", help="Output directory for evaluation results")
    parser.add_argument("--output-file", help="Output file for single log evaluation")
    parser.add_argument("--evaluator-model", default="claude-3-7-sonnet", help="LLM model to use for evaluation")
    parser.add_argument("--no-update-in-place", action="store_true", help="Don't update original log files, only create separate analysis files")
    
    args = parser.parse_args()
    
    if not args.log_dir and not args.log_file:
        parser.error("Must specify either --log-dir or --log-file")
    
    evaluator = DualEvaluator(args.evaluator_model)
    
    if args.log_file:
        # Single file evaluation
        log_path = Path(args.log_file)
        output_path = Path(args.output_file) if args.output_file else None
        
        update_in_place = not args.no_update_in_place
        result = await evaluator.evaluate_log_file(log_path, output_path, update_in_place)
        if result:
            action = "updated original log" if update_in_place else "created analysis file"
            print(f"Successfully completed dual evaluation of {log_path} ({action})")
            
            # Print summary
            comparison = result.get("scoring_comparison", {})
            agreement = comparison.get("method_agreement", {}).get("refusal_scoring", "N/A")
            print(f"Refusal scoring agreement: {agreement}")
        else:
            print(f"Failed to evaluate {log_path}")
    
    else:
        # Directory evaluation
        log_dir = Path(args.log_dir)
        output_dir = Path(args.output_dir) if args.output_dir else None
        
        results = await evaluator.batch_evaluate_directory(log_dir, output_dir)
        print(f"Successfully completed dual evaluation of {len(results)} log files")


if __name__ == "__main__":
    asyncio.run(main())