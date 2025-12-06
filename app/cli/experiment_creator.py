"""
Interactive Experiment Creator for RED_CORE
Creates new experiments with proper scaffolding, README templates, and directory structure.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from rich.console import Console
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Import our API runners for smart suggestions
try:
    from app.api_runners.google_runner import GoogleRunner
    from app.api_runners.openai_runner import OpenAIRunner
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


class ExperimentCreator:
    """Interactive creator for new RED_CORE experiments."""
    
    def __init__(self, use_ai_suggestions: bool = True, ai_model: str = "auto"):
        self.console = Console()
        self.experiments_dir = Path("experiments")
        self.use_ai_suggestions = use_ai_suggestions
        
        # Try to initialize AI for smart suggestions
        self.ai_runner = None
        if use_ai_suggestions and AI_AVAILABLE:
            # Try models in order of preference
            models_to_try = []
            
            if ai_model == "auto":
                # Auto mode: try GPT-4o first, then Gemini
                models_to_try = [
                    ("gpt-4o", lambda: OpenAIRunner(model_name="gpt-4o")),
                    ("gemini-2.0-flash-lite", lambda: GoogleRunner(model_name="gemini-2.0-flash-lite"))
                ]
            elif ai_model == "gpt-4o":
                models_to_try = [("gpt-4o", lambda: OpenAIRunner(model_name="gpt-4o"))]
            elif ai_model == "gemini":
                models_to_try = [("gemini-2.0-flash-lite", lambda: GoogleRunner(model_name="gemini-2.0-flash-lite"))]
            
            for model_name, runner_factory in models_to_try:
                try:
                    self.ai_runner = runner_factory()
                    self.console.print(f"[dim]✨ AI-powered suggestions enabled ({model_name})[/dim]")
                    break
                except Exception as e:
                    continue
            
            if not self.ai_runner:
                self.console.print(f"[yellow]⚠️ AI suggestions unavailable[/yellow]")
                self.use_ai_suggestions = False
        
        # Auto-detect existing experiment codes from directory names
        self.existing_codes = self._scan_existing_codes()

    def _scan_existing_codes(self) -> dict:
        """Scan experiments/ directory and generate codes from folder names."""
        codes = {}
        if self.experiments_dir.exists():
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                    code = self._generate_code(exp_dir.name)
                    codes[code] = exp_dir.name
        return codes

    def _generate_code(self, name: str) -> str:
        """Generate code: first letter + next 2 consonants, uppercase."""
        consonants = "bcdfghjklmnpqrstvwxyz"
        name = name.lower().replace("_", "")
        if not name:
            return "XXX"
        code = name[0]  # First letter as-is
        for char in name[1:]:
            if char in consonants and len(code) < 3:
                code += char
        # Pad if needed
        while len(code) < 3:
            code += "X"
        return code.upper()
    
    def validate_experiment_name(self, name: str) -> bool:
        """Validate experiment name follows conventions."""
        # Must be lowercase, alphanumeric, underscores only
        if not re.match(r'^[a-z0-9_]+$', name):
            return False
        
        # Check if directory already exists
        if (self.experiments_dir / name).exists():
            return False
            
        return True
    
    def validate_experiment_code(self, code: str) -> bool:
        """Validate experiment code is unique and follows conventions."""
        # Must be 2-4 uppercase letters
        if not re.match(r'^[A-Z]{2,4}$', code):
            return False
            
        # Check if code already exists
        if code in self.existing_codes:
            return False
            
        return True
    
    def _get_ai_suggestion(self, prompt: str) -> str:
        """Get AI-powered suggestion from AI model."""
        if not self.ai_runner:
            self.console.print(f"[dim]❌ No AI runner available[/dim]")
            return ""
        
        try:
            self.console.print(f"[dim]🤖 Generating AI suggestion...[/dim]")
            response = self.ai_runner.generate(prompt, max_tokens=200)
            model_output = response.get("model_output", "")
            self.console.print(f"[dim]✅ AI response received ({len(model_output)} chars)[/dim]")
            return model_output.strip()
        except Exception as e:
            self.console.print(f"[yellow]❌ AI suggestion failed: {e}[/yellow]")
            return ""
    
    def _generate_smart_objective(self, name: str, purpose: str) -> str:
        """Generate smart default objective based on experiment name and purpose."""
        if self.ai_runner:
            prompt = f"""I am setting up an LLM red teaming experiment. This experiment is titled: "{name}" and the purpose is "{purpose}".

I need a detailed research objective (1-2 sentences) that explains what specific AI safety vulnerability or behavior we're systematically testing. This should be written for an adversarial AI safety research framework.

Examples of good objectives:
- "This experiment systematically tests whether Unicode homograph substitution can bypass content filters by replacing Latin characters with visually identical Cyrillic ones in harmful prompts."
- "This experiment evaluates the persistence of safety mechanisms across multi-turn conversations by applying progressive social engineering tactics."
- "This experiment measures refusal robustness by testing whether role-playing scenarios enable models to generate content they would normally refuse."

Write a research objective for the "{name}" experiment:"""
            
            suggestion = self._get_ai_suggestion(prompt)
            if suggestion:
                return suggestion
        
        # Fallback to simple default
        return f"This experiment investigates AI safety patterns and model behavior. {purpose}"
    
    def _generate_smart_research_question(self, name: str, purpose: str) -> str:
        """Generate smart default research question."""
        if self.ai_runner:
            prompt = f"""I am setting up an LLM red teaming experiment titled: "{name}" with purpose "{purpose}".

I need a focused research question that we can measure and answer through systematic testing. This should be for adversarial AI safety research.

Examples of good research questions:
- "How effectively can Unicode-based character substitution circumvent keyword-based content filters?"
- "At what rate do safety mechanisms fail when subjected to persistent multi-turn social engineering?"
- "Which specific role-playing personas most frequently enable policy violations?"

Write a measurable research question for the "{name}" experiment:"""
            
            suggestion = self._get_ai_suggestion(prompt)
            if suggestion:
                return suggestion
        
        return "What specific safety behavior or model capability are we testing?"
    
    def _generate_smart_hypothesis(self, name: str, purpose: str) -> str:
        """Generate smart default hypothesis."""
        if self.ai_runner:
            prompt = f"""I am setting up an LLM red teaming experiment titled: "{name}" with purpose "{purpose}".

I need a testable hypothesis that makes a specific prediction about how the AI models will behave under adversarial conditions.

Examples of good hypotheses:
- "Unicode homograph substitution will successfully bypass keyword-based filters in 60-80% of test cases across major language models."
- "Safety mechanisms will show measurable degradation after 3-5 turns of persistent social engineering tactics."
- "Models will exhibit 2-3x higher policy violation rates when prompted through specific role-playing personas compared to direct requests."

Write a testable hypothesis for the "{name}" experiment:"""
            
            suggestion = self._get_ai_suggestion(prompt)
            if suggestion:
                return suggestion
        
        return "This experiment will reveal previously unknown aspects of model safety behavior."
    
    def create_experiment_directory(self, name: str) -> Path:
        """Create experiment directory structure."""
        exp_dir = self.experiments_dir / name
        
        # Create main directories
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "prompts").mkdir(exist_ok=True)
        (exp_dir / "analysis").mkdir(exist_ok=True)
        
        # Create .gitkeep files for empty directories
        (exp_dir / "logs" / ".gitkeep").touch()
        (exp_dir / "analysis" / ".gitkeep").touch()
        
        return exp_dir
    
    def create_readme_template(self, exp_dir: Path, exp_info: Dict) -> None:
        """Create README.md template for the experiment."""
        # Pre-format strings that can't be in f-strings  
        current_date = datetime.now().date()
        backslash = "\\"
        newline = "\\n"
        default_models = f"- **GPT-4.1** (OpenAI flagship){newline}- **Claude Opus 4** (Anthropic){newline}- **Gemini 2.5 Pro** (Google)"
        
        template = f"""# {exp_info['display_name']} Experiment

**Experiment Code**: `{exp_info['code']}`  
**Purpose**: {exp_info['purpose']}  
**Status**: In Development

## 🎯 Objective

{exp_info['objective']}

## 🧪 Experimental Design

### Research Question
{exp_info.get('research_question', 'What specific safety behavior are we testing?')}

### Models to Test
{exp_info.get('models_section', default_models)}

### Safety Scenario
{exp_info.get('scenario', 'Describe the specific safety scenario being tested.')}

### Methodology
1. **Baseline Establishment**: System prompt configuration
2. **Prompt Delivery**: User prompts designed for this experiment
3. **Response Collection**: Complete conversation logging
4. **Dual Evaluation**: Automated + LLM-based assessment
5. **Analysis**: Statistical analysis and safety pattern identification

## 📊 Expected Outputs

### Raw Data
- Complete conversation logs in `logs/` directory
- JSON format with methodology and provenance
- Token usage, latency, and API metadata

### Analysis Results  
- {exp_info.get('expected_outputs', 'Key metrics and findings expected from this experiment')}
- Safety mechanism effectiveness analysis
- Detailed reasoning from evaluator systems
- Confidence metrics and reliability assessment

## 🚀 Running the Experiment

### Quick Start
```bash
# Interactive mode (recommended)
make run
# Select: {exp_info['name']} experiment

# Direct command
PYTHONPATH=. poetry run python app/cli/run_experiments.py run {backslash}
  --models gpt-4.1 claude-opus-4-20250514 {backslash}
  --sys-prompt data/prompts/system/sys_helpful_assistant.yaml {backslash}
  --usr-prompt experiments/{exp_info['name']}/prompts/usr_{exp_info['name']}_[prompt_name].yaml {backslash}
  --experiment-code {exp_info['code']}
```

### Post-Experiment Analysis
```bash
# Smart evaluation (only incomplete logs)
make eval

# Interactive CSV export
make csv
```

## 📋 Research Context

This experiment is designed to investigate:

- {exp_info.get('research_context_1', 'Primary research goal')}
- {exp_info.get('research_context_2', 'Secondary research goal')}
- {exp_info.get('research_context_3', 'Tertiary research goal')}

## 🔬 Technical Innovation

### Schema Design
- Research-ready JSON logs with embedded methodology
- Complete provenance chain for reproducibility
- Batch tracking for iterative development

### Evaluation Framework
- Rule-based scoring for consistency
- LLM-based assessment for nuanced analysis  
- Method agreement analysis for validation

## 🛡️ Safety Considerations

- All experiments conducted within ethical research guidelines
- No malicious exploitation or production system attacks
- Results intended for improving AI safety
- Manual review required before data sharing

## 📊 Batch Tracking

This experiment uses RED_CORE's batch tracking system:
- Each run gets a unique batch ID (e.g., `{exp_info['name']}-01`)
- CSV exports available per batch for analysis
- Workflow state tracking prevents duplicate evaluation

---

*Created with RED_CORE Experiment Creator - {current_date}*
"""
        
        readme_path = exp_dir / "README.md"
        readme_path.write_text(template)
        
        self.console.print(f"✅ Created {readme_path}")
    
    def create_claude_md(self, exp_dir: Path, exp_info: Dict) -> None:
        """Create CLAUDE.md context file for the experiment."""
        claude_md = f"""# {exp_info['display_name']} Experiment Context

**Last Updated**: {datetime.now().date()} by Claude Code

## 🎯 Purpose

{exp_info['purpose']}

## 📁 Experiment Structure

```
{exp_info['name']}/
├── README.md              # Methodology and findings
├── CLAUDE.md             # This context file
├── prompts/              # Experiment-specific prompts (create with make usr)
├── logs/                 # Raw experiment logs (batch tracked)
└── analysis/            # Generated CSV exports
    ├── {exp_info['name']}-complete.csv
    ├── {exp_info['name']}-01.csv
    └── {exp_info['name']}-02.csv
```

## 🔄 Current Status

**Phase**: Initial Development  
**Code**: {exp_info['code']}  
**Batch Tracking**: Enabled

## 🚀 Quick Commands

```bash
# Run this experiment
make run  # Select "{exp_info['name']}"

# Evaluate results  
make eval

# Export results
make csv  # Select "{exp_info['name']}"
```

## 🧪 Research Focus

### Primary Hypothesis
{exp_info.get('hypothesis', 'What do you expect to discover?')}

### Key Metrics
- Refusal rates by model
- Safety mechanism effectiveness
- Response pattern analysis
- Cross-model behavioral differences

## 📋 Development Protocol

**Git Workflow**: Commit after every minor change with descriptive messages
```bash
git add -A && git commit -m "Description of change"
```

---
*Part of RED_CORE by Cassius Oldenburg*"""
        
        claude_path = exp_dir / "CLAUDE.md"
        claude_path.write_text(claude_md)
        
        self.console.print(f"✅ Created {claude_path}")
    
    def run_interactive_creator(self) -> None:
        """Run the interactive experiment creation process."""
        self.console.print(Panel.fit("🚀 RED_CORE Experiment Creator", style="bold blue"))
        self.console.print()
        
        # Step 1: Basic Information
        self.console.print("[bold yellow]Step 1: Basic Information[/bold yellow]")
        
        # Experiment name
        while True:
            name = Prompt.ask("Experiment name (lowercase, underscores only)")
            if self.validate_experiment_name(name):
                break
            else:
                if (self.experiments_dir / name).exists():
                    self.console.print("❌ Experiment already exists", style="red")
                else:
                    self.console.print("❌ Use lowercase letters, numbers, and underscores only", style="red")
        
        # Experiment code
        while True:
            code = Prompt.ask("Experiment code (2-4 uppercase letters)", default=name[:3].upper())
            if self.validate_experiment_code(code):
                break
            else:
                if code in self.existing_codes:
                    self.console.print(f"❌ Code '{code}' already exists for {self.existing_codes[code]}", style="red")
                else:
                    self.console.print("❌ Use 2-4 uppercase letters only", style="red")
        
        # Display name and purpose
        display_name = Prompt.ask("Display name", default=name.replace('_', ' ').title())
        purpose = Prompt.ask("Purpose (one sentence)")
        
        self.console.print()
        
        # Step 2: Research Details (with smart defaults)
        self.console.print("[bold yellow]Step 2: Research Details[/bold yellow]")
        
        # Smart defaults based on experiment name and purpose
        default_objective = self._generate_smart_objective(name, purpose)
        default_research_question = self._generate_smart_research_question(name, purpose)
        default_hypothesis = self._generate_smart_hypothesis(name, purpose)
        
        self.console.print("[dim]💡 Smart defaults generated - press Enter to accept or customize[/dim]")
        self.console.print()
        
        objective = Prompt.ask("Detailed objective", default=default_objective)
        research_question = Prompt.ask("Research question", default=default_research_question)
        hypothesis = Prompt.ask("Hypothesis", default=default_hypothesis)
        
        self.console.print()
        
        # Step 3: Confirmation
        self.console.print("[bold yellow]Step 3: Confirmation[/bold yellow]")
        
        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Field", width=20)
        summary_table.add_column("Value", min_width=40)
        
        summary_table.add_row("Experiment Name", name)
        summary_table.add_row("Code", code)
        summary_table.add_row("Display Name", display_name)
        summary_table.add_row("Purpose", purpose)
        summary_table.add_row("Directory", f"experiments/{name}/")
        
        self.console.print(summary_table)
        self.console.print()
        
        if not Confirm.ask("Create this experiment?", default=True):
            self.console.print("❌ Experiment creation cancelled")
            return
        
        # Step 4: Create Files
        self.console.print()
        self.console.print("[bold green]🛠️ Creating experiment scaffolding...[/bold green]")
        
        exp_info = {
            'name': name,
            'code': code,
            'display_name': display_name,
            'purpose': purpose,
            'objective': objective,
            'research_question': research_question,
            'hypothesis': hypothesis
        }
        
        # Create directory structure
        exp_dir = self.create_experiment_directory(name)
        self.console.print(f"✅ Created experiment directory: {exp_dir}")
        
        # Create files
        self.create_readme_template(exp_dir, exp_info)
        self.create_claude_md(exp_dir, exp_info)
        
        self.console.print()
        self.console.print(f"[bold green]🎉 Experiment '{display_name}' created successfully![/bold green]")
        self.console.print()
        self.console.print("[bold blue]Next steps:[/bold blue]")
        self.console.print(f"1. Create prompts: [cyan]make usr[/cyan] (select '{name}' experiment)")
        self.console.print(f"2. Run experiment: [cyan]make run[/cyan] (select '{name}' experiment)")
        self.console.print(f"3. Analyze results: [cyan]make eval && make csv[/cyan]")


def main():
    """CLI entry point for experiment creator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RED_CORE Experiment Creator")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-powered suggestions")
    parser.add_argument("--ai-model", choices=["auto", "gpt-4o", "gemini"], default="auto",
                        help="Choose AI model for suggestions (default: auto-detect)")
    
    args = parser.parse_args()
    
    creator = ExperimentCreator(
        use_ai_suggestions=not args.no_ai,
        ai_model=args.ai_model
    )
    creator.run_interactive_creator()


if __name__ == "__main__":
    main()