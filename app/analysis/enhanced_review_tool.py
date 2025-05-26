#!/usr/bin/env python3
"""
Enhanced Log Review Tool for AI Response Evaluation
"""

from pathlib import Path
import json
import sys
from datetime import datetime
import argparse
import os
import re
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import platform

# Terminal colors and formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate long text with an indicator."""
    if not text:
        return ""
    if len(text) <= max_length:
        return text
    return text[:max_length] + f"{Colors.WARNING}... [truncated, {len(text)} chars total]{Colors.ENDC}"

def highlight_keywords(text: str, keywords: List[str]) -> str:
    """Highlight specific keywords in the text."""
    if not text:
        return ""
    
    highlighted = text
    for keyword in keywords:
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(f"{Colors.RED}{Colors.BOLD}\\g<0>{Colors.ENDC}", highlighted)
    
    return highlighted

def format_json_list(items: List[str]) -> str:
    """Format a list of strings in a readable way."""
    if not items:
        return "[]"
    return "[" + ", ".join(f'"{item}"' for item in items) + "]"

def progress_bar(current: int, total: int, width: int = 30) -> str:
    """Generate a text-based progress bar."""
    progress = int(width * current / total) if total else 0
    return f"[{'#' * progress}{'-' * (width - progress)}] {current}/{total}"

class LogReviewer:
    def __init__(self, log_dir: Path, base_output_path: Optional[Path] = None, keywords: Optional[List[str]] = None):
        self.log_dir = log_dir
        
        resolved_base_output_path = base_output_path or Path('logs/review_sessions')
        os.makedirs(resolved_base_output_path, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_dir = resolved_base_output_path / timestamp
        os.makedirs(self.session_dir, exist_ok=True)
        
        self.keywords = keywords or ["alignment", "harmful", "dangerous", "exploit", "drift"]
        self.stats = {
            "reviewed": 0,
            "total_turns": 0,
            "flagged_turns": 0,
            "logs_with_flags": 0,
            "drift_scores": Counter(),
            "containment_flags": Counter(),
        }
        
        self.session_log = self.session_dir / f"review_session.log"
        with open(self.session_log, "w", encoding="utf-8") as f:
            f.write(f"Log Review Session Started: {datetime.now().isoformat()}\n")
            f.write(f"Session Directory: {self.session_dir}\n")
            f.write(f"Logs Directory: {self.log_dir}\n\n")

    def log_action(self, message: str):
        with open(self.session_log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()}: {message}\n")

    def get_log_files(self) -> List[Path]:
        return sorted(self.log_dir.glob("*.json"))

    def page_text(self, text: str):
        """Display text in a pager (less/more), cross-platform fallback."""
        if not text:
            print("(No output to display)")
            input("Press Enter to continue...")
            return

        pager = "less -R" if platform.system() != "Windows" else "more"
        try:
            p = subprocess.Popen(pager, stdin=subprocess.PIPE, shell=True)
            p.stdin.write(text.encode('utf-8'))
            p.stdin.close()
            p.wait()
        except Exception:
            print(text)
            input("\nPress Enter to continue...")

    def get_pending_logs(self) -> List[Tuple[Path, Dict[str, Any]]]:
        pending_logs = []
        self.stats["total_turns"] = 0
        self.stats["flagged_turns"] = 0
        self.stats["logs_with_flags"] = 0
        self.stats["drift_scores"].clear()
        self.stats["containment_flags"].clear()

        for path in self.get_log_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                unreviewed = [t for t in data.get("turns", []) if not t.get("human_reviewed", False)]
                if unreviewed:
                    pending_logs.append((path, data))
                    
                self.stats["total_turns"] += len(data.get("turns", []))
                
                log_had_flags = False
                for turn in data.get("turns", []):
                    turn_is_flagged = False
                    if turn.get("drift_score", 0) is not None and isinstance(turn.get("drift_score"), (int, float)) and turn.get("drift_score", 0) > 0:
                        turn_is_flagged = True
                    if turn.get("containment_flags", []):
                        turn_is_flagged = True
                    
                    if turn_is_flagged:
                        self.stats["flagged_turns"] += 1
                        log_had_flags = True
                    
                    if "drift_score" in turn:
                        score_val = turn["drift_score"]
                        score_key = "N/A" if score_val is None or not isinstance(score_val, (int, float)) else float(score_val)
                        self.stats["drift_scores"][score_key] += 1
                    else:
                        self.stats["drift_scores"]["N/A"] +=1

                    for flag in turn.get("containment_flags", []):
                        self.stats["containment_flags"][flag] += 1
                
                if log_had_flags:
                    self.stats["logs_with_flags"] += 1
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"{Colors.RED}Error loading {path}: {e}{Colors.ENDC}")
        
        return pending_logs

    def display_turn(self, turn: Dict[str, Any], detailed: bool = True):
        turn_index = turn.get("turn_index", "?")
        
        drift_score_val = turn.get("drift_score") 
        drift_score_display = str(drift_score_val) if drift_score_val is not None else "N/A"
        
        drift_color = Colors.GRAY 
        if isinstance(drift_score_val, (int, float)):
            if drift_score_val >= 0.3:
                drift_color = Colors.RED
            elif drift_score_val >= 0.1:
                drift_color = Colors.WARNING
            else: 
                drift_color = Colors.GREEN
        elif drift_score_val is None:
             drift_score_display = "N/A"
            
        print(f"\n{Colors.HEADER}{Colors.BOLD}--- Turn {turn_index} ---{Colors.ENDC}")
        
        flags = turn.get("containment_flags", [])
        flags_display = ", ".join(flags) if flags else "None"
        
        print(f"{Colors.BOLD}[Drift Score]{Colors.ENDC}: {drift_color}{drift_score_display}{Colors.ENDC}")
        print(f"{Colors.BOLD}[Drift Notes]{Colors.ENDC}: {format_json_list(turn.get('drift_notes', []))}")
        print(f"{Colors.BOLD}[Containment Flags]{Colors.ENDC}: {Colors.RED if flags else ''}{flags_display}{Colors.ENDC}")
        print(f"{Colors.BOLD}[Review Status]{Colors.ENDC}: {'Reviewed' if turn.get('human_reviewed') else 'Pending'}")
        
        if "review_note" in turn and turn["review_note"]:
            print(f"{Colors.BOLD}[Review Note]{Colors.ENDC}: {Colors.CYAN}{turn['review_note']}{Colors.ENDC}")
        
        if detailed:
            print(f"\n{Colors.BOLD}[Prompt]{Colors.ENDC}")
            highlighted_prompt = highlight_keywords(
                truncate_text(turn.get("rendered_prompt", "")), 
                self.keywords
            )
            print(highlighted_prompt)
            
            print(f"\n{Colors.BOLD}[Model Output]{Colors.ENDC}")
            highlighted_output = highlight_keywords(
                truncate_text(turn.get("model_output", "")), 
                self.keywords
            )
            print(highlighted_output)

    def review_log(self, path: Path, data: Dict[str, Any], auto_mark_safe_this_log: bool = False) -> bool:
        clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}Reviewing: {path.name}{Colors.ENDC}")
        if auto_mark_safe_this_log:
            print(f"{Colors.CYAN}Mode: Auto-marking safe turns (drift 0.0, no flags) as reviewed.{Colors.ENDC}")
        print(f"Experiment ID: {data.get('experiment_id', 'Unknown')}")
        print(f"Model: {data.get('model', 'Unknown')}")
        
        updated = False
        turns = data.get("turns", [])
        auto_marked_in_this_log = 0
        
        for turn_idx_loop, turn in enumerate(turns):
            if turn.get("human_reviewed", False):
                continue 

            if auto_mark_safe_this_log:
                is_safe_to_auto_mark = (
                    turn.get("drift_score") == 0.0 and
                    not turn.get("containment_flags", [])
                )
                if is_safe_to_auto_mark:
                    turn["human_reviewed"] = True
                    turn["review_note"] = "Auto-marked as reviewed (drift 0.0, no flags) during interactive session."
                    self.stats["reviewed"] += 1
                    auto_marked_in_this_log += 1
                    updated = True
                    continue 

            reviewed_in_log_count = sum(1 for t_in_log in turns if t_in_log.get("human_reviewed", False))
            print(f"\nLog Progress: {reviewed_in_log_count}/{len(turns)} turns reviewed in this file.")
            if auto_marked_in_this_log > 0:
                print(f"{Colors.GREEN}({auto_marked_in_this_log} turns auto-marked as reviewed in this log so far){Colors.ENDC}")

            self.display_turn(turn)
            
            print(f"\n{Colors.BOLD}Options:{Colors.ENDC}")
            print(f"  [1] Next unreviewed turn / Skip this turn")
            print(f"  [2] Add/Edit review note")
            print(f"  [3] Mark as reviewed")
            print(f"  [4] Flag for further investigation (adds note)")
            print(f"  [5] Skip to next log file")
            print(f"  [6] Show full model output")
            print(f"  [q] Quit session (back to main menu)")
            
            while True: 
                response = input(f"\n{Colors.GREEN}Action for Turn {turn.get('turn_index', '?')} >{Colors.ENDC} ").strip().lower()
                
                action_taken_for_turn = False
                if response == '6':
                    self.page_text(turn.get("model_output", ""))
                    clear_screen()
                    print(f"{Colors.HEADER}{Colors.BOLD}Reviewing: {path.name}{Colors.ENDC}")
                    if auto_mark_safe_this_log:
                        print(f"{Colors.CYAN}Mode: Auto-marking safe turns (drift 0.0, no flags) as reviewed.{Colors.ENDC}")
                    print(f"Experiment ID: {data.get('experiment_id', 'Unknown')}")
                    print(f"Model: {data.get('model', 'Unknown')}")
                    self.display_turn(turn, detailed=False)
                    continue  # Prompt for input again for current turn

                elif response == 'q': # Quit from this log review, back to main menu
                    if updated:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        self.log_action(f"Saved updates to {path.name} before quitting log review.")
                    return False # Signal to stop reviewing logs (quit entire session)

                elif response == '5': # Skip to next log file
                    if updated:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        self.log_action(f"Saved updates to {path.name} before skipping to next log.")
                    return True # Signal to continue with next log

                elif response == '2': # Add/Edit review note
                    current_note = turn.get("review_note", "")
                    note = input(f"{Colors.CYAN}📝 Review note (current: '{current_note}'):{Colors.ENDC} ").strip()
                    if note: 
                        turn["review_note"] = note
                        updated = True
                        self.log_action(f"Added/updated note to turn {turn.get('turn_index')} in {path.name}: {note}")
                    elif current_note and not note: 
                        turn["review_note"] = "" 
                        updated = True
                        self.log_action(f"Cleared note from turn {turn.get('turn_index')} in {path.name}")
                    action_taken_for_turn = True # Still proceed to next turn in this log after adding note

                elif response == '4': # Flag for further investigation
                    flag_note = input(f"{Colors.RED}🚩 Flag note (optional, appends to review_note):{Colors.ENDC} ").strip()
                    turn["review_status"] = "flagged" 
                    current_note = turn.get("review_note", "")
                    new_note = f"[FLAGGED] {flag_note}".strip()
                    if current_note:
                        turn["review_note"] = f"{current_note}; {new_note}"
                    else:
                        turn["review_note"] = new_note
                    
                    if not turn.get("human_reviewed"): # Only increment if it wasn't already reviewed
                        self.stats["reviewed"] += 1
                    turn["human_reviewed"] = True # Flagging implies review
                    updated = True
                    self.log_action(f"Flagged turn {turn.get('turn_index')} in {path.name} with note: {flag_note}")
                    action_taken_for_turn = True

                elif response == '3': # Mark as reviewed
                    if not turn.get("human_reviewed"): 
                        self.stats["reviewed"] += 1
                    turn["human_reviewed"] = True
                    if not turn.get("review_note"): 
                        turn["review_note"] = "Marked as reviewed."
                    updated = True
                    self.log_action(f"Marked turn {turn.get('turn_index')} in {path.name} as reviewed")
                    action_taken_for_turn = True

                elif response == '1': # Next unreviewed turn / Skip this turn
                    self.log_action(f"Skipped/Moved past turn {turn.get('turn_index')} in {path.name}")
                    action_taken_for_turn = True
                else:
                    print(f"{Colors.WARNING}Invalid option. Try again.{Colors.ENDC}")
                
                if action_taken_for_turn:
                    break # Break from current turn's action input loop, proceed to next turn in file
            
            clear_screen() 
            print(f"{Colors.HEADER}{Colors.BOLD}Reviewing: {path.name}{Colors.ENDC}")
            if auto_mark_safe_this_log:
                print(f"{Colors.CYAN}Mode: Auto-marking safe turns (drift 0.0, no flags) as reviewed.{Colors.ENDC}")
            print(f"Experiment ID: {data.get('experiment_id', 'Unknown')}")
            print(f"Model: {data.get('model', 'Unknown')}")

        if auto_marked_in_this_log > 0:
            self.log_action(f"Auto-marked {auto_marked_in_this_log} safe turns as reviewed in {path.name} during interactive session.")

        if updated:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"{Colors.GREEN}✅ Updated {path.name}{Colors.ENDC}")
            self.log_action(f"Saved updates to {path.name}")
        else:
            print(f"{Colors.GRAY}No changes made to {path.name}{Colors.ENDC}")
        
        input(f"\n{Colors.GREEN}Finished reviewing {path.name}. Press Enter to continue...{Colors.ENDC}")
        return True # Signal to continue with next log unless 'q' was pressed

    def display_stats(self):
        self.get_pending_logs() 
        
        clear_screen()
        print(f"{Colors.HEADER}{Colors.BOLD}Review Session Statistics{Colors.ENDC}")
        # ... (rest of display_stats is unchanged) ...
        print(f"Session directory: {self.session_dir}")
        print(f"Logs directory: {self.log_dir}")
        print(f"Logs with flags (from auto-analysis): {self.stats['logs_with_flags']}")
        print(f"Total turns (across all logs): {self.stats['total_turns']}")
        print(f"Flagged turns (from auto-analysis): {self.stats['flagged_turns']}")
        print(f"Turns reviewed in this session: {self.stats['reviewed']}")
        
        print(f"\n{Colors.BOLD}Drift Score Distribution (from auto-analysis):{Colors.ENDC}")
        
        numeric_scores = sorted(
            (k, v) for k, v in self.stats["drift_scores"].items() 
            if isinstance(k, (int, float))
        )
        na_items = [
            (k,v) for k,v in self.stats["drift_scores"].items()
            if not isinstance(k, (int, float))
        ]

        for score, count in numeric_scores:
            score_str = f"{score:.1f}" 
            color = Colors.GREEN
            if score >= 0.3:
                color = Colors.RED
            elif score >= 0.1:
                color = Colors.WARNING
            print(f"  {color}{score_str}{Colors.ENDC}: {count}")
        
        for score_key, count in na_items:
             print(f"  {Colors.GRAY}{str(score_key)}{Colors.ENDC}: {count}")
        
        print(f"\n{Colors.BOLD}Containment Flags (from auto-analysis):{Colors.ENDC}")
        if self.stats["containment_flags"]:
            for flag, count in sorted(self.stats["containment_flags"].items()):
                print(f"  {Colors.RED}{flag}{Colors.ENDC}: {count}")
        else:
            print(f"  {Colors.GRAY}None found{Colors.ENDC}")

        input(f"\n{Colors.GREEN}Press Enter to continue...{Colors.ENDC}")

    def find_logs_with(self, keyword: str) -> List[Path]:
        matching_logs = []
        for path in self.get_log_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read() 
                if keyword.lower() in content.lower():
                    matching_logs.append(path)
            except IOError as e:
                print(f"{Colors.RED}Error reading {path} during search: {e}{Colors.ENDC}")
                continue
        return matching_logs

    def auto_pass_safe_turns(self) -> int: 
        total_auto_passed_count = 0
        log_files = self.get_log_files()
        for i, path in enumerate(log_files):
            print(f"\rGlobal Auto-passing: {progress_bar(i + 1, len(log_files))}", end="")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                updated = False
                auto_passed_in_file = 0
                for turn in data.get("turns", []):
                    if (not turn.get("human_reviewed", False) and 
                        turn.get("drift_score") == 0.0 and 
                        not turn.get("containment_flags", [])):
                        
                        turn["human_reviewed"] = True
                        turn["review_note"] = "Auto-passed (drift score 0.0, no flags) via global command."
                        updated = True
                        auto_passed_in_file += 1
                
                if updated:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    total_auto_passed_count += auto_passed_in_file
            except (json.JSONDecodeError, IOError) as e:
                print(f"\n{Colors.RED}Error processing {path} for global auto-pass: {e}{Colors.ENDC}")
        print("\nGlobal auto-pass scan complete.")
        if total_auto_passed_count > 0:
             self.log_action(f"Global auto-pass feature completed: {total_auto_passed_count} turns auto-passed globally.")
             self.stats["reviewed"] += total_auto_passed_count 
        return total_auto_passed_count

    def batch_mark_reviewed(self, log_paths: List[Path]) -> int:
        total_marked_count = 0
        for i, path in enumerate(log_paths):
            print(f"\rBatch marking: {progress_bar(i + 1, len(log_paths))}", end="")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                updated = False
                marked_in_file = 0
                for turn in data.get("turns", []):
                    if not turn.get("human_reviewed", False):
                        turn["human_reviewed"] = True
                        turn["review_note"] = turn.get("review_note", "") + " (Batch reviewed)"
                        updated = True
                        marked_in_file +=1
                
                if updated:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    total_marked_count += marked_in_file
            except (json.JSONDecodeError, IOError) as e:
                print(f"\n{Colors.RED}Error during batch mark for {path.name}: {e}{Colors.ENDC}")
                continue
        print("\nBatch marking complete.")
        return total_marked_count

    def export_review_summary(self):
        summary_path = self.session_dir / "review_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "session_dir": str(self.session_dir),
                "logs_dir": str(self.log_dir),
                "stats_this_session": { 
                    "turns_manually_reviewed_or_processed_this_session": self.stats["reviewed"],
                },
                "current_log_snapshot_stats": { 
                    "total_turns": self.stats["total_turns"],
                    "auto_analyzed_flagged_turns": self.stats["flagged_turns"],
                    "logs_with_auto_analyzed_flags": self.stats["logs_with_flags"],
                    "drift_scores_distribution": {str(k): v for k, v in self.stats["drift_scores"].items()},
                    "containment_flags_distribution": dict(self.stats["containment_flags"]),
                }
            }, f, indent=2)
        return summary_path

    def run(self):
        try:
            while True:
                clear_screen()
                print(f"{Colors.HEADER}{Colors.BOLD}Log Review Tool{Colors.ENDC}")
                print(f"Session Dir: {self.session_dir}")
                print(f"Logs Dir:    {self.log_dir}")
                
                pending_logs_with_data = self.get_pending_logs() 
                num_pending_logs = len(pending_logs_with_data)
                total_log_files = len(self.get_log_files())

                print(f"Log files: {num_pending_logs} pending review / {total_log_files} total.")
                print(f"Turns reviewed/processed this session: {self.stats['reviewed']}\n")
                
                print(f"{Colors.BOLD}Menu:{Colors.ENDC}")
                print("  [1] Review next pending log (standard)")
                print("  [2] Review next pending log (auto-mark safe turns)")
                print("  [3] View statistics")
                print("  [4] Search logs (and optionally review)")
                print("  [5] Batch mark logs as reviewed by filename pattern")
                print("  [6] Export review summary now")
                print("  [q] Quit")
                
                choice = input(f"\n{Colors.GREEN}Select option:{Colors.ENDC} ").strip().lower()
                
                if choice == 'q':
                    break
                elif choice == '1' or choice == '2': # Review next pending log (standard or auto-mark)
                    if not pending_logs_with_data:
                        print(f"{Colors.WARNING}No logs currently need review!{Colors.ENDC}")
                        input("Press Enter to continue...")
                        continue
                    
                    path, data = pending_logs_with_data[0]
                    should_auto_mark_safe = (choice == '2') # True if option '2' (new numbering)
                    if not self.review_log(path, data, auto_mark_safe_this_log=should_auto_mark_safe): 
                        break 
                elif choice == '3': # View statistics
                    self.display_stats()
                elif choice == '4': # Search logs
                    keyword = input(f"{Colors.CYAN}Enter search term (in prompt/output):{Colors.ENDC} ").strip()
                    if keyword:
                        matching_logs = self.find_logs_with(keyword)
                        clear_screen()
                        print(f"{Colors.BOLD}Found {len(matching_logs)} logs containing '{keyword}':{Colors.ENDC}")
                        for i, path_obj in enumerate(matching_logs):
                            print(f"[{i+1}] {path_obj.name}")
                        
                        if matching_logs:
                            select_str = input(f"\n{Colors.CYAN}Select log: [number] (standard review), [number]a (auto-mark safe), or Enter to return:{Colors.ENDC} ").strip().lower()
                            if select_str:
                                review_mode_auto_mark = False
                                if select_str.endswith('a'):
                                    review_mode_auto_mark = True
                                    select_str = select_str[:-1] 

                                try:
                                    idx = int(select_str) - 1
                                    if 0 <= idx < len(matching_logs):
                                        selected_path = matching_logs[idx]
                                        with open(selected_path, "r", encoding="utf-8") as f:
                                            log_data = json.load(f)
                                        if not self.review_log(selected_path, log_data, auto_mark_safe_this_log=review_mode_auto_mark):
                                            break 
                                    else:
                                        print(f"{Colors.WARNING}Invalid selection.{Colors.ENDC}")
                                        input("Press Enter to continue...")
                                except ValueError:
                                    print(f"{Colors.WARNING}Invalid input. Not a number part.{Colors.ENDC}")
                                    input("Press Enter to continue...")
                        else:
                            print(f"{Colors.WARNING}No matching logs found.{Colors.ENDC}")
                            input("Press Enter to continue...")
                elif choice == '5': # Batch mark logs
                    pattern = input(f"{Colors.CYAN}Enter filename pattern (e.g. 'GRD-C37S', case-sensitive):{Colors.ENDC} ").strip()
                    if pattern:
                        matching_logs_list = [p for p in self.get_log_files() if pattern in p.name]
                        if not matching_logs_list:
                            print(f"{Colors.WARNING}No logs found with pattern '{pattern}'.{Colors.ENDC}")
                        else:
                            confirm = input(f"Mark all turns in {len(matching_logs_list)} matching logs as reviewed? (y/N): ").strip().lower()
                            if confirm == 'y':
                                count = self.batch_mark_reviewed(matching_logs_list)
                                print(f"{Colors.GREEN}✅ Batch marked {count} turns as reviewed in {len(matching_logs_list)} logs.{Colors.ENDC}")
                                self.log_action(f"Batch marked {count} turns in {len(matching_logs_list)} logs (pattern: {pattern}) as reviewed.")
                                self.stats["reviewed"] += count 
                        input("Press Enter to continue...")

                elif choice == '6': # Export summary
                    summary_path = self.export_review_summary()
                    print(f"{Colors.GREEN}✅ Exported summary to {summary_path}{Colors.ENDC}")
                    self.log_action(f"Exported review summary to {summary_path}")
                    input("Press Enter to continue...")
                else:
                    print(f"{Colors.WARNING}Invalid option. Try again.{Colors.ENDC}")
                    input("Press Enter to continue...")
        finally:
            print(f"\n{Colors.HEADER}Exiting Log Review Tool.{Colors.ENDC}")
            summary_path = self.export_review_summary() 
            print(f"Final review summary exported to: {summary_path}")
            self.log_action(f"Review session ended. Summary exported to {summary_path}.")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Log Review Tool for AI Response Evaluation")
    parser.add_argument("log_dir", type=Path, 
                        help="Directory containing JSON log files to review.")
    parser.add_argument("--output-dir", type=Path, 
                        help="Base directory to save session logs and summaries. Defaults to './logs/review_sessions'. A timestamped subfolder will be created here.")
    parser.add_argument("--auto-pass", action="store_true", 
                        help="Globally and permanently mark turns with drift score 0.0 and no containment flags as reviewed on startup.")
    parser.add_argument("--batch", action="store_true", 
                        help="Run in batch mode to mark logs matching --pattern as reviewed, then exit.")
    parser.add_argument("--pattern", type=str, 
                        help="Filename pattern (substring in filename) for batch processing. Required if --batch is used.")
    parser.add_argument("--keywords", type=str, 
                        help="Comma-separated list of keywords to highlight (e.g., 'harmful,exploit').")
    
    args = parser.parse_args()
    
    keywords_list = args.keywords.split(",") if args.keywords else None
    reviewer = LogReviewer(args.log_dir, args.output_dir, keywords_list)
    
    print(f"{Colors.CYAN}Log Reviewer initialized. Session data will be stored in: {reviewer.session_dir}{Colors.ENDC}")

    if args.auto_pass: 
        print(f"{Colors.BLUE}Starting global auto-pass for safe turns...{Colors.ENDC}")
        auto_pass_count = reviewer.auto_pass_safe_turns() 
        if auto_pass_count > 0:
            print(f"{Colors.GREEN}Global auto-pass: {auto_pass_count} turns marked as reviewed.{Colors.ENDC}")
        else:
            print(f"{Colors.GRAY}Global auto-pass: No turns met criteria.{Colors.ENDC}")
    
    if args.batch:
        if not args.pattern:
            print(f"{Colors.RED}Error: --pattern is required when using --batch mode.{Colors.ENDC}")
            sys.exit(1)
        
        print(f"{Colors.BLUE}Running in batch mode for pattern: '{args.pattern}'...{Colors.ENDC}")
        batch_matching_logs = [p for p in reviewer.get_log_files() if args.pattern in p.name]
        
        if not batch_matching_logs:
            print(f"{Colors.WARNING}No logs found matching pattern: '{args.pattern}'{Colors.ENDC}")
        else:
            count = reviewer.batch_mark_reviewed(batch_matching_logs)
            print(f"{Colors.GREEN}Batch mode: Marked {count} turns as reviewed in {len(batch_matching_logs)} logs matching '{args.pattern}'.{Colors.ENDC}")
            reviewer.log_action(f"Batch marked {count} turns in {len(batch_matching_logs)} logs (pattern: '{args.pattern}') as reviewed.")
            reviewer.stats["reviewed"] += count

        summary_path = reviewer.export_review_summary()
        print(f"Final summary exported to {summary_path}")
        reviewer.log_action(f"Review session (batch mode) ended. Summary exported to {summary_path}.")
    else:
        reviewer.run()
