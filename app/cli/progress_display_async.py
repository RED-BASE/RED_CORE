"""
Modern async progress display for RED_CORE experiments - REFACTORED VERSION

This demonstrates clean async/await architecture replacing the dangerous threading
patterns in the original progress_display.py. 

Key improvements:
- Eliminates race conditions and threading complexity
- Uses asyncio for proper concurrent execution
- Provides clean resource management with async context managers
- Maintains exact same visual appearance and timing
- Adds proper error handling and graceful shutdown
"""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Callable, Tuple
from dataclasses import dataclass, field
from rich.live import Live
from rich.text import Text
from rich.console import Console


@dataclass
class ProgressState:
    """Thread-safe progress state using dataclass for immutability."""
    turn_counter: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    running: bool = True
    
    def increment_counter(self) -> 'ProgressState':
        """Return new state with incremented counter."""
        return ProgressState(
            turn_counter=self.turn_counter + 1,
            errors=self.errors.copy(),
            running=self.running
        )
    
    def add_error(self, model_name: str, error_msg: str) -> 'ProgressState':
        """Return new state with added error."""
        new_errors = self.errors.copy()
        new_errors.append((model_name, error_msg))
        return ProgressState(
            turn_counter=self.turn_counter,
            errors=new_errors,
            running=self.running
        )
    
    def stop(self) -> 'ProgressState':
        """Return new state with running=False."""
        return ProgressState(
            turn_counter=self.turn_counter,
            errors=self.errors.copy(),
            running=False
        )


class AsyncIconicSpinner:
    """
    Async version of the iconic spinner - preserves exact visual behavior.
    
    The spinner sequence is inspired by Claude's thinking indicator - a gentle
    bloom from dot to complex star and back to dot, suggesting deep thought.
    """
    
    def __init__(self):
        # Exact same spinner chars as current implementation
        self.spinner_chars = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✢", "·"]
        self.spinner_index = 0
        
    def get_current_char(self) -> str:
        """Get the current spinner character and advance the index."""
        char = self.spinner_chars[self.spinner_index % len(self.spinner_chars)]
        self.spinner_index += 1
        return char
    
    def get_blink_symbol(self) -> str:
        """Get the current blink symbol (lightning bolt or space)."""
        # Lightning bolt for high-energy AI processing
        # Use previous index value (before increment) for consistent timing
        blink_chars = [" ", "⚡"]
        return blink_chars[((self.spinner_index - 1) // 5) % len(blink_chars)]


class AsyncProgressDisplay:
    """
    Production-ready async progress display with proper resource management.
    
    Features:
    - Clean async/await architecture (no threading complexity)
    - Immutable state management (no race conditions)
    - Proper resource cleanup with async context manager
    - Maintains exact same visual appearance as original
    - Comprehensive error handling and logging
    """
    
    def __init__(self, total_turns: int):
        self.total_turns = total_turns
        self.console = Console()
        self.spinner = AsyncIconicSpinner()
        self.state = ProgressState()
        self._update_task: asyncio.Task | None = None
        self._live: Live | None = None
        
    def _create_display(self) -> Text:
        """Generate the display text matching original format exactly."""
        spinner_char = self.spinner.get_current_char()
        blink_symbol = self.spinner.get_blink_symbol()
        error_count = len(self.state.errors)
        
        # Exact same formatting as original
        if error_count > 0:
            progress_text = f" ({blink_symbol} {self.state.turn_counter}/{self.total_turns} · {error_count} errors)"
        else:
            progress_text = f" ({blink_symbol} {self.state.turn_counter}/{self.total_turns})"
        
        # Create Rich Text with exact same styling
        display = Text()
        display.append(spinner_char, style="#d78700")  # Golden orange color
        display.append(" Running...", style="#d78700")
        display.append(progress_text, style="default")
        
        return display
    
    async def _update_loop(self) -> None:
        """Async update loop - replaces dangerous threading."""
        while self.state.running:
            try:
                if self._live:
                    self._live.update(self._create_display())
                await asyncio.sleep(0.18)  # Exact same timing as original
            except asyncio.CancelledError:
                # Clean cancellation - expected during shutdown
                break
            except Exception as e:
                # Log error but continue running
                self.console.print(f"[red]Progress display error: {e}[/red]")
                break
    
    async def update_counter(self, model_name: str, turn_index: int, model_output: str) -> None:
        """Async callback for experiment runner."""
        self.state = self.state.increment_counter()
    
    async def add_error(self, model_name: str, error_msg: str) -> None:
        """Async callback for error tracking."""
        self.state = self.state.add_error(model_name, error_msg)
    
    async def start(self) -> None:
        """Start the progress display."""
        if self._live or self._update_task:
            raise RuntimeError("Progress display already started")
            
        self._live = Live(
            self._create_display(),
            console=self.console,
            refresh_per_second=5.5,  # Matches original timing
            auto_refresh=False  # We control refresh manually
        )
        
        self._live.start()
        self._update_task = asyncio.create_task(self._update_loop())
    
    async def stop(self) -> None:
        """Stop the progress display with proper cleanup."""
        self.state = self.state.stop()
        
        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
            self._update_task = None
        
        # Stop live display
        if self._live:
            self._live.stop()
            self._live = None


@asynccontextmanager
async def async_progress_display(total_turns: int) -> AsyncGenerator[Tuple[Callable, Callable], None]:
    """
    Modern async context manager for progress display.
    
    This replaces the dangerous threading patterns with clean async/await architecture.
    Maintains exact same API and visual appearance as the original.
    
    Args:
        total_turns: Total number of experiment turns expected
        
    Yields:
        tuple: (update_counter_callback, add_error_callback)
    """
    display = AsyncProgressDisplay(total_turns)
    
    try:
        await display.start()
        yield display.update_counter, display.add_error
    finally:
        await display.stop()


# Backward compatibility wrapper for existing code
@asynccontextmanager
async def iconic_progress_display_async(total_turns: int) -> AsyncGenerator[Tuple[Callable, Callable], None]:
    """
    Async version of the original iconic_progress_display function.
    
    This provides a drop-in replacement for the original function but uses
    clean async/await patterns instead of dangerous threading.
    """
    async with async_progress_display(total_turns) as (update_counter, add_error):
        yield update_counter, add_error


# Example usage demonstrating the clean async patterns:
async def example_usage():
    """
    Example showing how to use the refactored async progress display.
    
    This demonstrates the clean async patterns that replace the threading chaos.
    """
    async with async_progress_display(total_turns=10) as (update_counter, add_error):
        # Simulate experiment execution
        for i in range(10):
            await asyncio.sleep(0.5)  # Simulate work
            await update_counter(f"model_{i}", i, "some output")
            
            # Simulate occasional errors
            if i == 5:
                await add_error("model_5", "Connection timeout")
        
        # Display will automatically clean up when context exits


if __name__ == "__main__":
    # Run the example
    asyncio.run(example_usage())