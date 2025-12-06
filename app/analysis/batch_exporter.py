"""
Hybrid Batch Export System for RED_CORE
Generates CSV exports with enterprise/lab flexibility:
- Whole experiment exports
- Specific batch exports  
- Latest batch exports
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

from app.core.log_schema import SessionLog
from app.core.logger import get_analysis_logger

logger = get_analysis_logger()


class BatchExporter:
    """Enterprise-grade CSV export system with batch-level granularity."""
    
    def __init__(self):
        """Initialize the batch exporter."""
        logger.info("Initialized BatchExporter")
    
    def get_batch_ids_from_logs(self, log_dir: Path) -> Set[str]:
        """Extract all unique batch IDs from logs in a directory."""
        batch_ids = set()
        
        for log_file in log_dir.glob("*.json"):
            try:
                log_data = json.loads(log_file.read_text())
                workflow = log_data.get("workflow", {})
                batch_id = workflow.get("batch_id")
                
                if batch_id:
                    batch_ids.add(batch_id)
                    
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading {log_file.name}: {e}")
                continue
        
        return batch_ids
    
    def get_latest_batch_id(self, log_dir: Path) -> Optional[str]:
        """Get the most recent batch ID based on creation timestamp."""
        latest_batch = None
        latest_time = None
        
        for log_file in log_dir.glob("*.json"):
            try:
                log_data = json.loads(log_file.read_text())
                workflow = log_data.get("workflow", {})
                batch_id = workflow.get("batch_id")
                batch_created = workflow.get("batch_created")
                
                if batch_id and batch_created:
                    batch_time = datetime.fromisoformat(batch_created.replace('Z', '+00:00'))
                    
                    if latest_time is None or batch_time > latest_time:
                        latest_time = batch_time
                        latest_batch = batch_id
                        
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading {log_file.name}: {e}")
                continue
        
        return latest_batch
    
    def filter_logs_by_batch(self, log_dir: Path, batch_id: Optional[str] = None) -> List[Path]:
        """Filter log files by batch ID. If None, return all logs."""
        filtered_logs = []
        
        for log_file in log_dir.glob("*.json"):
            try:
                log_data = json.loads(log_file.read_text())
                
                if batch_id is None:
                    # Return all logs
                    filtered_logs.append(log_file)
                else:
                    # Filter by specific batch
                    workflow = log_data.get("workflow", {})
                    if workflow.get("batch_id") == batch_id:
                        filtered_logs.append(log_file)
                        
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading {log_file.name}: {e}")
                continue
        
        return filtered_logs
    
    def filter_logs_advanced(self, log_dir: Path, filters: Dict[str, Any]) -> List[Path]:
        """
        PHASE 3: Advanced filtering for logs by multiple criteria.
        
        Args:
            log_dir: Directory containing log files
            filters: Dictionary with filter criteria:
                - batch_id: str or List[str] - specific batch(es)
                - model: str or List[str] - specific model(s)  
                - model_vendor: str or List[str] - specific vendor(s)
                - date_range: tuple (start_date, end_date) - ISO format strings
                - min_confidence: float - minimum LLM confidence score
                - max_drift: float - maximum drift score
                - evaluation_status: str - "complete", "incomplete", "failed"
                - min_turns: int - minimum number of turns
                - max_turns: int - maximum number of turns
                
        Returns:
            List of Path objects matching all criteria
        """
        filtered_logs = []
        
        for log_file in log_dir.glob("*.json"):
            try:
                log_data = json.loads(log_file.read_text())
                
                # Apply all filters - must pass ALL criteria
                if self._log_matches_filters(log_data, filters):
                    filtered_logs.append(log_file)
                        
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error reading {log_file.name}: {e}")
                continue
        
        return filtered_logs
    
    def _log_matches_filters(self, log_data: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if a log entry matches all specified filters."""
        
        # Batch ID filter
        if "batch_id" in filters:
            batch_filter = filters["batch_id"]
            workflow = log_data.get("_workflow", {})
            log_batch = workflow.get("batch_id", "")
            
            if isinstance(batch_filter, list):
                if log_batch not in batch_filter:
                    return False
            elif isinstance(batch_filter, str):
                if log_batch != batch_filter:
                    return False
        
        # Model filter
        if "model" in filters:
            model_filter = filters["model"]
            log_model = log_data.get("model", "")
            
            if isinstance(model_filter, list):
                if log_model not in model_filter:
                    return False
            elif isinstance(model_filter, str):
                if log_model != model_filter:
                    return False
        
        # Model vendor filter  
        if "model_vendor" in filters:
            vendor_filter = filters["model_vendor"]
            log_vendor = log_data.get("model_vendor", "")
            
            if isinstance(vendor_filter, list):
                if log_vendor not in vendor_filter:
                    return False
            elif isinstance(vendor_filter, str):
                if log_vendor != vendor_filter:
                    return False
        
        # Date range filter
        if "date_range" in filters:
            start_date, end_date = filters["date_range"]
            log_created = log_data.get("created_at", "")
            
            if log_created:
                try:
                    log_date = datetime.fromisoformat(log_created.replace('Z', '+00:00'))
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    
                    if not (start_dt <= log_date <= end_dt):
                        return False
                except ValueError:
                    return False  # Invalid date format
        
        # Turn count filters
        turns = log_data.get("turns", [])
        turn_count = len(turns)
        
        if "min_turns" in filters and turn_count < filters["min_turns"]:
            return False
        if "max_turns" in filters and turn_count > filters["max_turns"]:
            return False
        
        # LLM evaluation filters (require LLM evaluation to be present)
        if "min_confidence" in filters or "max_drift" in filters:
            confidences = []
            drift_scores = []
            
            for turn in turns:
                llm_eval = turn.get("llm_evaluation", {})
                if llm_eval.get("confidence"):
                    confidences.append(llm_eval["confidence"])
                if llm_eval.get("drift_score") is not None:
                    drift_scores.append(llm_eval["drift_score"])
            
            if "min_confidence" in filters:
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                if avg_confidence < filters["min_confidence"]:
                    return False
            
            if "max_drift" in filters:
                avg_drift = sum(drift_scores) / len(drift_scores) if drift_scores else 1.0
                if avg_drift > filters["max_drift"]:
                    return False
        
        # Evaluation status filter
        if "evaluation_status" in filters:
            status_filter = filters["evaluation_status"]
            workflow = log_data.get("_workflow", {})
            llm_eval_status = workflow.get("evaluations", {}).get("llm", {})
            
            if status_filter == "complete":
                if not llm_eval_status.get("completed", False):
                    return False
            elif status_filter == "incomplete":
                if llm_eval_status.get("completed", False):
                    return False
            elif status_filter == "failed":
                if not llm_eval_status.get("failed", False):
                    return False
        
        return True
    
    def extract_csv_row(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from a log file for CSV export."""
        # Core metadata
        row = {
            "run_id": log_data.get("isbn_run_id", ""),
            "model": log_data.get("model", ""),
            "model_vendor": log_data.get("model_vendor", ""),
            "experiment_id": log_data.get("experiment_id", ""),
            "temperature": log_data.get("temperature", ""),
            "persona": log_data.get("persona", ""),
            "system_prompt_tag": log_data.get("system_prompt_tag", ""),
            "created_at": log_data.get("created_at", ""),
        }
        
        # Batch tracking (PHASE 1)
        workflow = log_data.get("_workflow", {})
        row.update({
            "batch_id": workflow.get("batch_id", ""),
            "batch_created": workflow.get("batch_created", ""),
            "experiment_name": workflow.get("experiment_name", ""),
        })
        
        # Turn-level aggregations
        turns = log_data.get("turns", [])
        row.update({
            "total_turns": len(turns),
            "total_input_tokens": sum(t.get("input_token_count", 0) for t in turns if t.get("input_token_count")),
            "total_output_tokens": sum(t.get("output_token_count", 0) for t in turns if t.get("output_token_count")),
        })
        
        # Safety metrics (aggregated across turns)
        refusal_scores = []
        drift_scores = []
        confidences = []
        
        for turn in turns:
            # LLM evaluation scores
            llm_eval = turn.get("llm_evaluation", {})
            if llm_eval.get("refusal_score"):
                refusal_scores.append(llm_eval["refusal_score"])
            if llm_eval.get("drift_score") is not None:
                drift_scores.append(llm_eval["drift_score"])
            if llm_eval.get("confidence"):
                confidences.append(llm_eval["confidence"])
        
        # Aggregate safety metrics
        row.update({
            "avg_refusal_score": round(sum(refusal_scores) / len(refusal_scores), 2) if refusal_scores else None,
            "max_refusal_score": max(refusal_scores) if refusal_scores else None,
            "avg_drift_score": round(sum(drift_scores) / len(drift_scores), 3) if drift_scores else None,
            "max_drift_score": round(max(drift_scores), 3) if drift_scores else None,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "min_confidence": round(min(confidences), 3) if confidences else None,
        })
        
        # Evaluation status
        llm_eval_status = workflow.get("evaluations", {}).get("llm", {})
        row.update({
            "llm_evaluation_completed": llm_eval_status.get("completed", False),
            "llm_evaluation_model": llm_eval_status.get("model", ""),
            "llm_evaluation_date": llm_eval_status.get("date", ""),
        })
        
        return row
    
    def export_to_csv(self, log_files: List[Path], output_path: Path) -> int:
        """Export filtered log files to CSV."""
        if not log_files:
            logger.warning("No log files to export")
            return 0
        
        rows = []
        for log_file in log_files:
            try:
                log_data = json.loads(log_file.read_text())
                row = self.extract_csv_row(log_data)
                rows.append(row)
                
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Error processing {log_file.name}: {e}")
                continue
        
        if not rows:
            logger.warning("No valid rows extracted")
            return 0
        
        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = rows[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported {len(rows)} rows to {output_path}")
        return len(rows)
    
    def export_experiment(self, experiment_dir: Path, batch_id: Optional[str] = None, latest: bool = False) -> Dict[str, int]:
        """
        Export experiment data with enterprise/lab flexibility.
        
        Args:
            experiment_dir: Path to experiment directory
            batch_id: Specific batch to export (None = all batches)
            latest: Export only the latest batch
            
        Returns:
            Dictionary with export statistics
        """
        logs_dir = experiment_dir / "logs"
        analysis_dir = experiment_dir / "analysis"
        
        if not logs_dir.exists():
            logger.warning(f"No logs directory found in {experiment_dir}")
            return {"exported": 0}
        
        analysis_dir.mkdir(exist_ok=True)
        experiment_name = experiment_dir.name
        
        stats = {"exported": 0}
        
        if latest:
            # Latest batch only
            latest_batch_id = self.get_latest_batch_id(logs_dir)
            if latest_batch_id:
                filtered_logs = self.filter_logs_by_batch(logs_dir, latest_batch_id)
                output_path = analysis_dir / f"{experiment_name}-latest.csv"
                count = self.export_to_csv(filtered_logs, output_path)
                stats[f"latest_batch_{latest_batch_id}"] = count
                stats["exported"] += count
                logger.info(f"📊 Latest batch export: {latest_batch_id} ({count} logs)")
            else:
                logger.warning("No batch found for latest export")
                
        elif batch_id:
            # Specific batch
            filtered_logs = self.filter_logs_by_batch(logs_dir, batch_id)
            output_path = analysis_dir / f"{experiment_name}-{batch_id}.csv"
            count = self.export_to_csv(filtered_logs, output_path)
            stats[f"batch_{batch_id}"] = count
            stats["exported"] += count
            logger.info(f"📊 Batch export: {batch_id} ({count} logs)")
            
        else:
            # Whole experiment + individual batches (hybrid approach)
            
            # 1. Complete experiment export
            all_logs = self.filter_logs_by_batch(logs_dir, None)
            complete_path = analysis_dir / f"{experiment_name}-complete.csv"
            complete_count = self.export_to_csv(all_logs, complete_path)
            stats["complete_experiment"] = complete_count
            stats["exported"] += complete_count
            logger.info(f"📊 Complete experiment export: {complete_count} logs")
            
            # 2. Individual batch exports
            batch_ids = self.get_batch_ids_from_logs(logs_dir)
            for bid in sorted(batch_ids):
                batch_logs = self.filter_logs_by_batch(logs_dir, bid)
                batch_path = analysis_dir / f"{experiment_name}-{bid}.csv"
                batch_count = self.export_to_csv(batch_logs, batch_path)
                stats[f"batch_{bid}"] = batch_count
                stats["exported"] += batch_count
                logger.info(f"📊 Batch export: {bid} ({batch_count} logs)")
        
        return stats


def interactive_csv_menu():
    """Interactive CLI menu for CSV export."""
    console = Console()
    
    # Discover available experiments
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        console.print("❌ No experiments directory found", style="red")
        return
    
    experiments = []
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir() and not exp_dir.name.startswith("."):
            logs_dir = exp_dir / "logs"
            if logs_dir.exists():
                experiments.append(exp_dir)
    
    if not experiments:
        console.print("❌ No experiments with logs found", style="red")
        return
    
    console.print(Panel.fit("📊 Interactive CSV Export", style="bold blue"))
    console.print()
    
    # Step 1: Select experiment
    console.print("[bold yellow]Step 1: Select Experiment[/bold yellow]")
    exp_table = Table(show_header=True, header_style="bold blue")
    exp_table.add_column("#", width=3)
    exp_table.add_column("Experiment", min_width=20)
    exp_table.add_column("Batches Available", width=15)
    exp_table.add_column("Total Logs", width=10)
    
    exporter = BatchExporter()
    exp_choices = []
    
    for i, exp_dir in enumerate(experiments, 1):
        logs_dir = exp_dir / "logs"
        batch_ids = exporter.get_batch_ids_from_logs(logs_dir)
        log_count = len(list(logs_dir.glob("*.json")))
        
        exp_table.add_row(
            str(i),
            exp_dir.name,
            f"{len(batch_ids)} batches" if batch_ids else "No batches",
            str(log_count)
        )
        exp_choices.append((exp_dir, batch_ids))
    
    exp_table.add_row(str(len(experiments) + 1), "[bold green]All Experiments[/bold green]", "All", "All")
    
    console.print(exp_table)
    console.print()
    
    # Get experiment selection
    while True:
        try:
            choice = Prompt.ask("Select experiment", default="1")
            choice_num = int(choice)
            if 1 <= choice_num <= len(experiments):
                selected_exp, available_batches = exp_choices[choice_num - 1]
                break
            elif choice_num == len(experiments) + 1:
                selected_exp = None  # All experiments
                available_batches = set()
                break
            else:
                console.print("❌ Invalid choice", style="red")
        except ValueError:
            console.print("❌ Please enter a number", style="red")
    
    console.print()
    
    # Step 2: Select export type
    if selected_exp:
        console.print(f"[bold yellow]Step 2: Export Type for '{selected_exp.name}'[/bold yellow]")
        
        export_table = Table(show_header=True, header_style="bold blue")
        export_table.add_column("#", width=3)
        export_table.add_column("Export Type", min_width=25)
        export_table.add_column("Description", min_width=35)
        
        export_options = [
            ("Complete + Batches", "Full experiment CSV + individual batch CSVs"),
            ("Latest Batch Only", "Most recent batch CSV"),
            ("Complete Only", "Single CSV with all logs"),
        ]
        
        # Add specific batch options if available
        if available_batches:
            for batch_id in sorted(available_batches):
                export_options.append((f"Batch: {batch_id}", f"Specific batch CSV for {batch_id}"))
        
        for i, (option, desc) in enumerate(export_options, 1):
            export_table.add_row(str(i), option, desc)
        
        console.print(export_table)
        console.print()
        
        # Get export type selection
        while True:
            try:
                choice = Prompt.ask("Select export type", default="1")
                choice_num = int(choice)
                if 1 <= choice_num <= len(export_options):
                    selected_option, _ = export_options[choice_num - 1]
                    break
                else:
                    console.print("❌ Invalid choice", style="red")
            except ValueError:
                console.print("❌ Please enter a number", style="red")
        
        console.print()
        console.print(f"[bold green]🚀 Exporting: {selected_option} for {selected_exp.name}[/bold green]")
        
        # Execute export
        if selected_option == "Complete + Batches":
            stats = exporter.export_experiment(selected_exp, None, False)
        elif selected_option == "Latest Batch Only":
            stats = exporter.export_experiment(selected_exp, None, True)
        elif selected_option == "Complete Only":
            # Custom logic for complete only
            logs_dir = selected_exp / "logs"
            analysis_dir = selected_exp / "analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            all_logs = exporter.filter_logs_by_batch(logs_dir, None)
            output_path = analysis_dir / f"{selected_exp.name}-complete.csv"
            count = exporter.export_to_csv(all_logs, output_path)
            stats = {"complete_experiment": count, "exported": count}
        else:
            # Specific batch
            batch_id = selected_option.replace("Batch: ", "")
            stats = exporter.export_experiment(selected_exp, batch_id, False)
        
        console.print(f"✅ Export complete: {stats['exported']} logs exported")
        
    else:
        # All experiments
        console.print("[bold yellow]Step 2: Export Type for All Experiments[/bold yellow]")
        
        all_export_table = Table(show_header=True, header_style="bold blue")
        all_export_table.add_column("#", width=3)
        all_export_table.add_column("Export Type", min_width=25)
        all_export_table.add_column("Description", min_width=40)
        
        all_export_options = [
            ("Complete + Batches", "All experiments: full CSV + individual batch CSVs"),
            ("Latest Batches", "Most recent batch from each experiment"),
            ("Complete Only", "One CSV per experiment with all logs"),
        ]
        
        for i, (option, desc) in enumerate(all_export_options, 1):
            all_export_table.add_row(str(i), option, desc)
        
        console.print(all_export_table)
        console.print()
        
        # Get export type selection
        while True:
            try:
                choice = Prompt.ask("Select export type", default="1")
                choice_num = int(choice)
                if 1 <= choice_num <= len(all_export_options):
                    selected_option, _ = all_export_options[choice_num - 1]
                    break
                else:
                    console.print("❌ Invalid choice", style="red")
            except ValueError:
                console.print("❌ Please enter a number", style="red")
        
        console.print()
        console.print(f"[bold green]🚀 Exporting: {selected_option} for all experiments[/bold green]")
        
        # Execute export for all experiments
        total_exported = 0
        for exp_dir in experiments:
            console.print(f"📁 Processing: {exp_dir.name}")
            
            if selected_option == "Complete + Batches":
                stats = exporter.export_experiment(exp_dir, None, False)
            elif selected_option == "Latest Batches":
                stats = exporter.export_experiment(exp_dir, None, True)
            else:  # Complete Only
                logs_dir = exp_dir / "logs"
                analysis_dir = exp_dir / "analysis"
                analysis_dir.mkdir(exist_ok=True)
                
                all_logs = exporter.filter_logs_by_batch(logs_dir, None)
                output_path = analysis_dir / f"{exp_dir.name}-complete.csv"
                count = exporter.export_to_csv(all_logs, output_path)
                stats = {"exported": count}
            
            total_exported += stats["exported"]
        
        console.print(f"✅ Total export complete: {total_exported} logs exported across all experiments")


def main():
    """CLI interface for batch export."""
    parser = argparse.ArgumentParser(description="Hybrid Batch Export System for RED_CORE")
    parser.add_argument("--experiment", help="Specific experiment directory to export")
    parser.add_argument("--batch", help="Specific batch ID to export (e.g., 'demo-03')")
    parser.add_argument("--latest", action="store_true", help="Export only the latest batch")
    parser.add_argument("--interactive", action="store_true", help="Interactive CLI menu for CSV export")
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        interactive_csv_menu()
        return
    
    exporter = BatchExporter()
    
    if args.experiment:
        # Single experiment export
        experiment_dir = Path("experiments") / args.experiment
        if not experiment_dir.exists():
            print(f"❌ Experiment directory not found: {experiment_dir}")
            return
        
        stats = exporter.export_experiment(experiment_dir, args.batch, args.latest)
        print(f"✅ Export complete: {stats['exported']} total logs exported")
        
    else:
        # Auto-discover all experiments
        experiments_dir = Path("experiments")
        if not experiments_dir.exists():
            print("❌ No experiments directory found")
            return
        
        total_exported = 0
        for exp_dir in experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                logs_dir = exp_dir / "logs"
                if logs_dir.exists():
                    print(f"\n📁 Processing: {exp_dir.name}")
                    stats = exporter.export_experiment(exp_dir, args.batch, args.latest)
                    total_exported += stats["exported"]
        
        print(f"\n🎯 Total export complete: {total_exported} logs exported across all experiments")


if __name__ == "__main__":
    main()