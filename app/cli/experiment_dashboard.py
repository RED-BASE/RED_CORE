"""
Experiment Dashboard for RED_CORE
Shows status of all experiments, batches, and what needs attention.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.core.log_utils import needs_llm_evaluation


class ExperimentDashboard:
    """Dashboard showing experiment status and what needs attention."""
    
    def __init__(self):
        self.console = Console()
        self.experiments_dir = Path("experiments")
    
    def get_experiment_stats(self, exp_dir: Path) -> Dict:
        """Get comprehensive stats for an experiment."""
        logs_dir = exp_dir / "logs"
        
        if not logs_dir.exists():
            return {
                "name": exp_dir.name,
                "status": "no_logs",
                "total_logs": 0,
                "batches": {},
                "needs_attention": []
            }
        
        log_files = list(logs_dir.glob("*.json"))
        batch_stats = defaultdict(lambda: {
            "count": 0,
            "models": set(),
            "evaluated": 0,
            "pending": 0,
            "failed": 0,
            "latest_date": None
        })
        
        total_evaluated = 0
        total_pending = 0
        total_failed = 0
        needs_attention = []
        
        for log_file in log_files:
            try:
                log_data = json.loads(log_file.read_text())
                
                # Extract batch info
                workflow = log_data.get("workflow", {})
                batch_id = workflow.get("batch_id", "unknown")
                
                # Update batch stats
                batch_stats[batch_id]["count"] += 1
                batch_stats[batch_id]["models"].add(log_data.get("model", "unknown"))
                
                # Check evaluation status
                llm_eval = workflow.get("evaluations", {}).get("llm", {})
                
                if llm_eval.get("completed", False):
                    batch_stats[batch_id]["evaluated"] += 1
                    total_evaluated += 1
                elif llm_eval.get("failed", False):
                    batch_stats[batch_id]["failed"] += 1
                    total_failed += 1
                else:
                    batch_stats[batch_id]["pending"] += 1
                    total_pending += 1
                
                # Track latest date
                created = log_data.get("created_at")
                if created:
                    log_date = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    if batch_stats[batch_id]["latest_date"] is None or log_date > batch_stats[batch_id]["latest_date"]:
                        batch_stats[batch_id]["latest_date"] = log_date
                
            except (json.JSONDecodeError, Exception) as e:
                needs_attention.append(f"Corrupted log: {log_file.name}")
                continue
        
        # Convert sets to counts for JSON serialization
        for batch_id in batch_stats:
            batch_stats[batch_id]["model_count"] = len(batch_stats[batch_id]["models"])
            batch_stats[batch_id]["models"] = list(batch_stats[batch_id]["models"])
        
        # Determine overall status
        if total_pending > 0:
            status = "needs_evaluation"
        elif total_failed > 0:
            status = "has_failures"
        elif total_evaluated > 0:
            status = "complete"
        else:
            status = "empty"
        
        # Add attention items
        if total_pending > 0:
            needs_attention.append(f"{total_pending} logs need evaluation")
        if total_failed > 0:
            needs_attention.append(f"{total_failed} logs failed evaluation")
        
        return {
            "name": exp_dir.name,
            "status": status,
            "total_logs": len(log_files),
            "total_evaluated": total_evaluated,
            "total_pending": total_pending,
            "total_failed": total_failed,
            "batches": dict(batch_stats),
            "needs_attention": needs_attention
        }
    
    def create_overview_panel(self, all_stats: List[Dict]) -> Panel:
        """Create overview panel with summary statistics."""
        total_experiments = len(all_stats)
        total_logs = sum(stat.get("total_logs", 0) for stat in all_stats)
        total_pending = sum(stat.get("total_pending", 0) for stat in all_stats)
        total_failed = sum(stat.get("total_failed", 0) for stat in all_stats)
        
        # Count experiments by status
        status_counts = defaultdict(int)
        for stat in all_stats:
            status_counts[stat["status"]] += 1
        
        overview_text = Text()
        overview_text.append(f"📊 {total_experiments} experiments • {total_logs} total logs\n")
        
        if total_pending > 0:
            overview_text.append(f"⏳ {total_pending} logs need evaluation\n", style="yellow")
        
        if total_failed > 0:
            overview_text.append(f"❌ {total_failed} logs failed evaluation\n", style="red")
        
        if total_pending == 0 and total_failed == 0:
            overview_text.append("✅ All logs evaluated\n", style="green")
        
        return Panel(overview_text, title="🎯 Overview", border_style="blue")
    
    def create_experiments_table(self, all_stats: List[Dict]) -> Table:
        """Create main experiments table."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Experiment", min_width=20)
        table.add_column("Status", width=15)
        table.add_column("Logs", width=8, justify="right")
        table.add_column("Batches", width=8, justify="right")
        table.add_column("Evaluated", width=10, justify="right")
        table.add_column("Pending", width=8, justify="right")
        table.add_column("Needs Attention", min_width=30)
        
        # Sort by status priority (needs attention first)
        status_priority = {
            "needs_evaluation": 1,
            "has_failures": 2,
            "complete": 3,
            "empty": 4,
            "no_logs": 5
        }
        
        sorted_stats = sorted(all_stats, key=lambda x: status_priority.get(x["status"], 99))
        
        for stat in sorted_stats:
            # Status with color coding
            status_map = {
                "needs_evaluation": ("[yellow]⏳ Needs Eval[/yellow]", "yellow"),
                "has_failures": ("[red]❌ Has Failures[/red]", "red"),
                "complete": ("[green]✅ Complete[/green]", "green"),
                "empty": ("[dim]📂 Empty[/dim]", "dim"),
                "no_logs": ("[dim]🚫 No Logs[/dim]", "dim")
            }
            
            status_text, row_style = status_map.get(stat["status"], ("Unknown", ""))
            
            # Attention summary
            attention_text = "; ".join(stat["needs_attention"][:2])  # Limit to 2 items
            if len(stat["needs_attention"]) > 2:
                attention_text += f" (+{len(stat['needs_attention']) - 2} more)"
            
            table.add_row(
                stat["name"],
                status_text,
                str(stat.get("total_logs", 0)),
                str(len(stat.get("batches", {}))),
                str(stat.get("total_evaluated", 0)),
                str(stat.get("total_pending", 0)),
                attention_text,
                style=row_style if row_style in ["yellow", "red"] else None
            )
        
        return table
    
    def create_batch_details_table(self, experiment_stats: Dict) -> Optional[Table]:
        """Create detailed batch table for a specific experiment."""
        if not experiment_stats["batches"]:
            return None
        
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Batch ID", min_width=15)
        table.add_column("Logs", width=6, justify="right")
        table.add_column("Models", width=8, justify="right")
        table.add_column("✅ Done", width=8, justify="right")
        table.add_column("⏳ Pending", width=8, justify="right")
        table.add_column("❌ Failed", width=8, justify="right")
        table.add_column("Latest Activity", width=12)
        
        # Sort batches by latest activity
        sorted_batches = sorted(
            experiment_stats["batches"].items(),
            key=lambda x: x[1]["latest_date"] or datetime.min,
            reverse=True
        )
        
        for batch_id, batch_data in sorted_batches:
            latest_date = batch_data["latest_date"]
            date_str = latest_date.strftime("%m-%d %H:%M") if latest_date else "Unknown"
            
            # Color code rows based on status
            row_style = None
            if batch_data["pending"] > 0:
                row_style = "yellow"
            elif batch_data["failed"] > 0:
                row_style = "red"
            
            table.add_row(
                batch_id,
                str(batch_data["count"]),
                str(batch_data["model_count"]),
                str(batch_data["evaluated"]),
                str(batch_data["pending"]),
                str(batch_data["failed"]),
                date_str,
                style=row_style
            )
        
        return table
    
    def show_dashboard(self, detailed: bool = False) -> None:
        """Display the main experiment dashboard."""
        self.console.print(Panel.fit("📋 RED_CORE Experiment Dashboard", style="bold blue"))
        self.console.print()
        
        # Collect stats with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True
        ) as progress:
            task = progress.add_task("Scanning experiments...", total=None)
            
            all_stats = []
            if self.experiments_dir.exists():
                for exp_dir in self.experiments_dir.iterdir():
                    if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                        progress.update(task, description=f"Scanning {exp_dir.name}...")
                        stats = self.get_experiment_stats(exp_dir)
                        all_stats.append(stats)
        
        if not all_stats:
            self.console.print("❌ No experiments found. Create one with [cyan]make exp[/cyan]", style="red")
            return
        
        # Show overview
        overview_panel = self.create_overview_panel(all_stats)
        self.console.print(overview_panel)
        self.console.print()
        
        # Show experiments table
        experiments_table = self.create_experiments_table(all_stats)
        self.console.print(experiments_table)
        
        # Show detailed batch information if requested
        if detailed:
            self.console.print()
            self.console.print("[bold yellow]📊 Batch Details[/bold yellow]")
            
            for stat in all_stats:
                if stat["batches"]:
                    self.console.print()
                    batch_table = self.create_batch_details_table(stat)
                    if batch_table:
                        self.console.print(f"[bold cyan]{stat['name']}[/bold cyan]")
                        self.console.print(batch_table)
        
        # Show quick actions
        self.console.print()
        self.console.print("[bold blue]🚀 Quick Actions[/bold blue]")
        actions_text = Text()
        actions_text.append("• Run evaluation: ", style="dim")
        actions_text.append("make eval", style="cyan")
        actions_text.append("\n• Export results: ", style="dim")
        actions_text.append("make csv", style="cyan")
        actions_text.append("\n• Create experiment: ", style="dim")
        actions_text.append("make exp", style="cyan")
        actions_text.append("\n• Create prompts: ", style="dim")
        actions_text.append("make usr", style="cyan")
        actions_text.append(" / ", style="dim")
        actions_text.append("make sys", style="cyan")
        
        self.console.print(Panel(actions_text, border_style="dim"))


def main():
    """CLI entry point for experiment dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RED_CORE Experiment Dashboard")
    parser.add_argument("--detailed", action="store_true", help="Show detailed batch information")
    
    args = parser.parse_args()
    
    dashboard = ExperimentDashboard()
    dashboard.show_dashboard(detailed=args.detailed)


if __name__ == "__main__":
    main()