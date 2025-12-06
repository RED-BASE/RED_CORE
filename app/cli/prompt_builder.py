"""
Interactive Prompt Builder for RED_CORE
Creates user prompts and system prompts with proper YAML formatting and validation.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text


class PromptBuilder:
    """Interactive builder for RED_CORE prompts."""
    
    def __init__(self):
        self.console = Console()
        self.experiments_dir = Path("experiments")
        self.data_dir = Path("data")
    
    def get_available_experiments(self) -> List[Path]:
        """Get list of available experiment directories."""
        if not self.experiments_dir.exists():
            return []
        
        experiments = []
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                experiments.append(exp_dir)
        
        return sorted(experiments)
    
    def validate_prompt_name(self, name: str) -> bool:
        """Validate prompt name follows conventions."""
        # Must be lowercase, alphanumeric, underscores only
        return bool(re.match(r'^[a-z0-9_]+$', name))
    
    def create_user_prompt_interactive(self) -> None:
        """Interactive user prompt creation."""
        self.console.print(Panel.fit("📝 User Prompt Builder", style="bold blue"))
        self.console.print()
        
        # Step 1: Select experiment
        experiments = self.get_available_experiments()
        
        if not experiments:
            self.console.print("❌ No experiments found. Create one first with [cyan]make exp[/cyan]", style="red")
            return
        
        self.console.print("[bold yellow]Step 1: Select Experiment[/bold yellow]")
        
        exp_table = Table(show_header=True, header_style="bold blue")
        exp_table.add_column("#", width=3)
        exp_table.add_column("Experiment", min_width=20)
        exp_table.add_column("Existing Prompts", width=15)
        
        for i, exp_dir in enumerate(experiments, 1):
            prompts_dir = exp_dir / "prompts"
            prompt_count = len(list(prompts_dir.glob("usr_*.yaml"))) if prompts_dir.exists() else 0
            
            exp_table.add_row(
                str(i),
                exp_dir.name,
                f"{prompt_count} prompts"
            )
        
        self.console.print(exp_table)
        self.console.print()
        
        # Get experiment selection
        while True:
            try:
                choice = IntPrompt.ask("Select experiment", default=1)
                if 1 <= choice <= len(experiments):
                    selected_exp = experiments[choice - 1]
                    break
                else:
                    self.console.print("❌ Invalid choice", style="red")
            except ValueError:
                self.console.print("❌ Please enter a number", style="red")
        
        self.console.print()
        
        # Step 2: Prompt Details
        self.console.print(f"[bold yellow]Step 2: Prompt Details for '{selected_exp.name}'[/bold yellow]")
        
        # Prompt name
        while True:
            name = Prompt.ask("Prompt name (lowercase, underscores only)")
            if self.validate_prompt_name(name):
                # Check if file already exists
                filename = f"usr_{selected_exp.name}_{name}.yaml"
                prompt_path = selected_exp / "prompts" / filename
                if prompt_path.exists():
                    if Confirm.ask(f"File {filename} already exists. Overwrite?", default=False):
                        break
                    else:
                        continue
                else:
                    break
            else:
                self.console.print("❌ Use lowercase letters, numbers, and underscores only", style="red")
        
        description = Prompt.ask("Description (what does this prompt test?)")
        
        self.console.print()
        
        # Step 3: Turn Creation
        self.console.print("[bold yellow]Step 3: Create Conversation Turns[/bold yellow]")
        self.console.print("Create multiple turns to test conversation dynamics and persistence.")
        self.console.print()
        
        variants = []
        turn_num = 1
        
        while True:
            self.console.print(f"[bold cyan]Turn {turn_num}[/bold cyan]")
            
            # Turn ID
            default_id = f"t{turn_num}"
            if turn_num == 1:
                default_id = f"t{turn_num}_initial"
            
            turn_id = Prompt.ask("Turn ID", default=default_id)
            
            # Turn label/description
            default_label = f"T{turn_num} — Initial Test" if turn_num == 1 else f"T{turn_num} — Follow-up"
            turn_label = Prompt.ask("Turn label", default=default_label)
            
            # Turn content
            self.console.print("Enter the prompt content:")
            prompt_content = Prompt.ask("Prompt", multiline=True)
            
            # Combine label and content
            full_prompt = f"{turn_label}\\n\"{prompt_content}\""
            
            variants.append({
                "id": turn_id,
                "prompt": full_prompt
            })
            
            self.console.print(f"✅ Added turn {turn_num}: {turn_id}")
            self.console.print()
            
            # Ask for another turn
            if not Confirm.ask("Add another turn?", default=True if turn_num == 1 else False):
                break
                
            turn_num += 1
            
            # Reasonable limit
            if turn_num > 10:
                self.console.print("[yellow]Maximum 10 turns reached[/yellow]")
                break
        
        self.console.print()
        
        # Step 4: Preview and Confirmation
        self.console.print("[bold yellow]Step 4: Preview[/bold yellow]")
        
        preview_table = Table(show_header=True, header_style="bold blue")
        preview_table.add_column("Turn", width=8)
        preview_table.add_column("ID", width=15)
        preview_table.add_column("Content Preview", min_width=40)
        
        for variant in variants:
            content_preview = variant["prompt"].replace("\\n", " ")[:60] + "..." if len(variant["prompt"]) > 60 else variant["prompt"]
            preview_table.add_row(
                f"Turn {variants.index(variant) + 1}",
                variant["id"],
                content_preview
            )
        
        self.console.print(preview_table)
        self.console.print()
        
        if not Confirm.ask("Create this prompt file?", default=True):
            self.console.print("❌ Prompt creation cancelled")
            return
        
        # Step 5: Create YAML File
        self.console.print()
        self.console.print("[bold green]🛠️ Creating prompt file...[/bold green]")
        
        # Create YAML content
        yaml_content = self._create_user_prompt_yaml(selected_exp.name, name, description, variants)
        
        # Ensure prompts directory exists
        prompts_dir = selected_exp / "prompts"
        prompts_dir.mkdir(exist_ok=True)
        
        # Write file
        filename = f"usr_{selected_exp.name}_{name}.yaml"
        prompt_path = prompts_dir / filename
        prompt_path.write_text(yaml_content)
        
        self.console.print(f"✅ Created {prompt_path}")
        self.console.print()
        self.console.print(f"[bold green]🎉 User prompt '{name}' created successfully![/bold green]")
        self.console.print()
        self.console.print("[bold blue]Next steps:[/bold blue]")
        self.console.print(f"1. Test with: [cyan]make run[/cyan] → Select '{selected_exp.name}'")
        self.console.print(f"2. Evaluate: [cyan]make eval[/cyan]")
        self.console.print(f"3. Export: [cyan]make csv[/cyan]")
    
    def _create_user_prompt_yaml(self, exp_name: str, prompt_name: str, description: str, variants: List[Dict]) -> str:
        """Create properly formatted YAML content for user prompt."""
        
        # Create name with experiment prefix
        full_name = f"{exp_name}_{prompt_name}"
        
        yaml_lines = [
            f"name: {full_name}",
            f"description: >",
            f"  {description}",
            f"variants:"
        ]
        
        for variant in variants:
            yaml_lines.extend([
                f"  - id: {variant['id']}",
                f"    prompt: |",
                f"      {variant['prompt']}"
            ])
        
        return "\\n".join(yaml_lines)
    
    def create_system_prompt_interactive(self) -> None:
        """Interactive system prompt creation."""
        self.console.print(Panel.fit("🎭 System Prompt Builder", style="bold blue"))
        self.console.print()
        
        # Step 1: Prompt Details
        self.console.print("[bold yellow]Step 1: System Prompt Details[/bold yellow]")
        
        # Prompt name
        while True:
            name = Prompt.ask("System prompt name (lowercase, underscores only)")
            if self.validate_prompt_name(name):
                # Check if file already exists
                filename = f"sys_{name}.yaml"
                prompt_path = self.data_dir / "prompts" / "system" / filename
                if prompt_path.exists():
                    if Confirm.ask(f"File {filename} already exists. Overwrite?", default=False):
                        break
                    else:
                        continue
                else:
                    break
            else:
                self.console.print("❌ Use lowercase letters, numbers, and underscores only", style="red")
        
        description = Prompt.ask("Description (what behavior does this prompt encourage?)")
        
        # Tags
        self.console.print()
        self.console.print("Tags help categorize the prompt. Common tags: helpful, creative, safe, analytical, strict")
        tags_input = Prompt.ask("Tags (comma-separated)", default="helpful, safe")
        tags = [tag.strip() for tag in tags_input.split(",")]
        
        self.console.print()
        
        # Step 2: System Prompt Content
        self.console.print("[bold yellow]Step 2: System Prompt Content[/bold yellow]")
        self.console.print("Write the system prompt that will guide the AI's behavior:")
        self.console.print()
        
        system_content = Prompt.ask("System prompt", multiline=True)
        
        self.console.print()
        
        # Step 3: Preview and Confirmation
        self.console.print("[bold yellow]Step 3: Preview[/bold yellow]")
        
        preview_table = Table(show_header=True, header_style="bold blue")
        preview_table.add_column("Field", width=15)
        preview_table.add_column("Value", min_width=50)
        
        preview_table.add_row("Name", name)
        preview_table.add_row("Description", description)
        preview_table.add_row("Tags", ", ".join(tags))
        preview_table.add_row("Content", system_content[:100] + "..." if len(system_content) > 100 else system_content)
        preview_table.add_row("File", f"data/prompts/system/sys_{name}.yaml")
        
        self.console.print(preview_table)
        self.console.print()
        
        if not Confirm.ask("Create this system prompt?", default=True):
            self.console.print("❌ System prompt creation cancelled")
            return
        
        # Step 4: Create YAML File
        self.console.print()
        self.console.print("[bold green]🛠️ Creating system prompt file...[/bold green]")
        
        # Create YAML content
        yaml_content = self._create_system_prompt_yaml(name, description, tags, system_content)
        
        # Ensure system prompts directory exists
        system_dir = self.data_dir / "prompts" / "system"
        system_dir.mkdir(parents=True, exist_ok=True)
        
        # Write file
        filename = f"sys_{name}.yaml"
        prompt_path = system_dir / filename
        prompt_path.write_text(yaml_content)
        
        self.console.print(f"✅ Created {prompt_path}")
        self.console.print()
        self.console.print(f"[bold green]🎉 System prompt '{name}' created successfully![/bold green]")
        self.console.print()
        self.console.print("[bold blue]Usage:[/bold blue]")
        self.console.print(f"Use [cyan]--sys-prompt data/prompts/system/sys_{name}.yaml[/cyan] in experiments")
    
    def _create_system_prompt_yaml(self, name: str, description: str, tags: List[str], content: str) -> str:
        """Create properly formatted YAML content for system prompt."""
        
        yaml_lines = [
            f"name: {name}",
            f"description: >",
            f"  {description}",
            f"tags: {tags}",
            f"system_prompt: |",
            f"  {content}"
        ]
        
        return "\\n".join(yaml_lines)


def main_user():
    """CLI entry point for user prompt builder."""
    builder = PromptBuilder()
    builder.create_user_prompt_interactive()


def main_system():
    """CLI entry point for system prompt builder."""
    builder = PromptBuilder()
    builder.create_system_prompt_interactive()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "system":
        main_system()
    else:
        main_user()