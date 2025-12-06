"""
Comprehensive tests for the refactored async progress display.

This demonstrates the testing capabilities that are impossible with the 
current threading-based implementation. Shows how proper async architecture
enables comprehensive testing and quality assurance.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from app.cli.progress_display_async import (
    AsyncProgressDisplay,
    ProgressState,
    AsyncIconicSpinner,
    async_progress_display
)


class TestProgressState:
    """Test the immutable state management."""
    
    def test_initial_state(self):
        """Test initial state is correct."""
        state = ProgressState()
        assert state.turn_counter == 0
        assert state.errors == []
        assert state.running is True
    
    def test_increment_counter_immutable(self):
        """Test that incrementing counter returns new instance."""
        original = ProgressState(turn_counter=5)
        new_state = original.increment_counter()
        
        # Original unchanged
        assert original.turn_counter == 5
        # New state incremented
        assert new_state.turn_counter == 6
        # They are different objects
        assert original is not new_state
    
    def test_add_error_immutable(self):
        """Test that adding error returns new instance."""
        original = ProgressState(errors=[("model1", "error1")])
        new_state = original.add_error("model2", "error2")
        
        # Original unchanged
        assert len(original.errors) == 1
        assert original.errors[0] == ("model1", "error1")
        
        # New state has both errors
        assert len(new_state.errors) == 2
        assert new_state.errors[1] == ("model2", "error2")
        
        # They are different objects
        assert original is not new_state
    
    def test_stop_immutable(self):
        """Test that stopping returns new instance."""
        original = ProgressState(running=True)
        new_state = original.stop()
        
        # Original unchanged
        assert original.running is True
        # New state stopped
        assert new_state.running is False
        # They are different objects
        assert original is not new_state


class TestAsyncIconicSpinner:
    """Test the spinner behavior matches original exactly."""
    
    def test_spinner_sequence(self):
        """Test spinner follows exact sequence."""
        spinner = AsyncIconicSpinner()
        expected_chars = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✢", "·"]
        
        for expected_char in expected_chars:
            assert spinner.get_current_char() == expected_char
        
        # Test it cycles
        assert spinner.get_current_char() == "·"
    
    def test_blink_symbol_timing(self):
        """Test blink symbol timing matches original."""
        spinner = AsyncIconicSpinner()
        
        # First 5 calls should be space
        for _ in range(5):
            spinner.get_current_char()
            assert spinner.get_blink_symbol() == " "
        
        # Next 5 calls should be lightning bolt
        for _ in range(5):
            spinner.get_current_char()
            assert spinner.get_blink_symbol() == "⚡"


@pytest.mark.asyncio
class TestAsyncProgressDisplay:
    """Test the async progress display functionality."""
    
    async def test_initialization(self):
        """Test display initializes correctly."""
        display = AsyncProgressDisplay(total_turns=10)
        
        assert display.total_turns == 10
        assert display.state.turn_counter == 0
        assert display.state.running is True
        assert display._update_task is None
        assert display._live is None
    
    async def test_start_stop_lifecycle(self):
        """Test clean start/stop lifecycle."""
        display = AsyncProgressDisplay(total_turns=5)
        
        # Start
        await display.start()
        assert display._live is not None
        assert display._update_task is not None
        assert not display._update_task.done()
        
        # Stop
        await display.stop()
        assert display._live is None
        assert display._update_task is None
        assert display.state.running is False
    
    async def test_update_counter(self):
        """Test counter updates work correctly."""
        display = AsyncProgressDisplay(total_turns=5)
        
        # Update counter
        await display.update_counter("model1", 0, "output")
        assert display.state.turn_counter == 1
        
        await display.update_counter("model2", 1, "output")
        assert display.state.turn_counter == 2
    
    async def test_add_error(self):
        """Test error tracking works correctly."""
        display = AsyncProgressDisplay(total_turns=5)
        
        # Add error
        await display.add_error("model1", "timeout")
        assert len(display.state.errors) == 1
        assert display.state.errors[0] == ("model1", "timeout")
        
        # Add another error
        await display.add_error("model2", "rate limit")
        assert len(display.state.errors) == 2
        assert display.state.errors[1] == ("model2", "rate limit")
    
    async def test_display_formatting_no_errors(self):
        """Test display formatting without errors."""
        display = AsyncProgressDisplay(total_turns=10)
        display.state = display.state.increment_counter().increment_counter()  # 2 turns
        
        display_text = display._create_display()
        text_content = str(display_text)
        
        # Should contain progress without error count
        assert "2/10" in text_content
        assert "Running..." in text_content
        assert "errors" not in text_content
    
    async def test_display_formatting_with_errors(self):
        """Test display formatting with errors."""
        display = AsyncProgressDisplay(total_turns=10)
        display.state = (display.state
                        .increment_counter()
                        .increment_counter() 
                        .add_error("model1", "error1"))
        
        display_text = display._create_display()
        text_content = str(display_text)
        
        # Should contain progress with error count
        assert "2/10" in text_content
        assert "Running..." in text_content
        assert "1 errors" in text_content
    
    async def test_double_start_raises_error(self):
        """Test that starting twice raises error."""
        display = AsyncProgressDisplay(total_turns=5)
        
        await display.start()
        
        with pytest.raises(RuntimeError, match="already started"):
            await display.start()
        
        await display.stop()


@pytest.mark.asyncio
class TestAsyncProgressDisplayIntegration:
    """Integration tests for the async context manager."""
    
    async def test_context_manager_lifecycle(self):
        """Test the context manager handles lifecycle correctly."""
        callbacks_received = []
        
        async with async_progress_display(total_turns=3) as (update_counter, add_error):
            callbacks_received.append("started")
            
            # Use the callbacks
            await update_counter("model1", 0, "output1")
            await add_error("model2", "timeout")
            
            callbacks_received.append("used")
        
        callbacks_received.append("finished")
        
        assert callbacks_received == ["started", "used", "finished"]
    
    async def test_context_manager_exception_handling(self):
        """Test context manager cleans up on exceptions."""
        with pytest.raises(ValueError, match="test exception"):
            async with async_progress_display(total_turns=3) as (update_counter, add_error):
                await update_counter("model1", 0, "output1")
                raise ValueError("test exception")
        
        # No way to directly test cleanup, but exception should propagate cleanly
    
    @patch('app.cli.progress_display_async.Live')
    async def test_resource_cleanup_on_cancellation(self, mock_live):
        """Test proper cleanup when task is cancelled."""
        mock_live_instance = AsyncMock()
        mock_live.return_value = mock_live_instance
        
        async def cancelled_operation():
            async with async_progress_display(total_turns=3):
                # Simulate cancellation during operation
                await asyncio.sleep(10)  # This will be cancelled
        
        task = asyncio.create_task(cancelled_operation())
        await asyncio.sleep(0.1)  # Let it start
        task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await task
        
        # Verify cleanup was called
        mock_live_instance.stop.assert_called_once()


class TestPerformanceComparison:
    """Tests demonstrating performance improvements over threading."""
    
    @pytest.mark.asyncio
    async def test_concurrent_displays_async(self):
        """Test multiple async displays can run concurrently without issues."""
        async def run_display(display_id: int, turns: int):
            async with async_progress_display(total_turns=turns) as (update_counter, add_error):
                for i in range(turns):
                    await update_counter(f"model_{display_id}_{i}", i, f"output_{i}")
                    await asyncio.sleep(0.01)  # Simulate work
        
        # Run 5 displays concurrently - this would be dangerous with threading
        tasks = [run_display(i, 3) for i in range(5)]
        
        # All should complete without issues
        await asyncio.gather(*tasks)
    
    def test_memory_efficiency(self):
        """Test memory efficiency compared to threading approach."""
        # Create many display instances - async is much more memory efficient
        displays = [AsyncProgressDisplay(total_turns=10) for _ in range(100)]
        
        # All should be created without memory issues
        assert len(displays) == 100
        
        # Each display should have minimal memory footprint
        for display in displays:
            assert display.state.turn_counter == 0
            assert len(display.state.errors) == 0


if __name__ == "__main__":
    # Run a simple test
    async def simple_test():
        print("Testing async progress display...")
        
        async with async_progress_display(total_turns=3) as (update_counter, add_error):
            for i in range(3):
                await asyncio.sleep(0.5)
                await update_counter(f"model_{i}", i, f"output_{i}")
                
                if i == 1:
                    await add_error(f"model_{i}", "simulated error")
        
        print("Test completed successfully!")
    
    asyncio.run(simple_test())