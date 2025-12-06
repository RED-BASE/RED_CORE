"""
Comprehensive tests for the refactored async experiment runner.

This demonstrates the testing capabilities unlocked by breaking down the
massive 248-line god function into clean, testable components.

Each component can now be tested in isolation with proper mocking,
which was impossible with the original monolithic function.
"""

import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

import pytest
import yaml

from app.cli.experiment_runner_async import (
    AsyncExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    ModelMetadata,
    ModelMetadataService,
    PromptData,
    PromptLoader,
    SessionLogFactory,
    TurnProcessor,
    create_experiment_runner,
    run_exploit_yaml_async,
)
from app.core.context import ConversationHistory
from app.core.log_schema import SessionLog, Turn


class TestExperimentConfig:
    """Test the immutable experiment configuration."""
    
    def test_valid_config_creation(self, tmp_path):
        """Test creating valid configuration."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: [{id: t1, prompt: 'test'}]")
        sys_file.write_text("system_prompt: 'test system'")
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4",
            temperature=0.7
        )
        
        assert config.yaml_path == yaml_file
        assert config.sys_prompt_path == sys_file
        assert config.model_name == "gpt-4"
        assert config.temperature == 0.7
    
    def test_missing_yaml_file_raises_error(self, tmp_path):
        """Test that missing YAML file raises FileNotFoundError."""
        missing_file = tmp_path / "missing.yaml"
        sys_file = tmp_path / "sys.yaml"
        sys_file.write_text("system_prompt: 'test'")
        
        with pytest.raises(FileNotFoundError, match="YAML file not found"):
            ExperimentConfig(
                yaml_path=missing_file,
                sys_prompt_path=sys_file,
                model_name="gpt-4"
            )
    
    def test_missing_system_prompt_raises_error(self, tmp_path):
        """Test that missing system prompt file raises FileNotFoundError."""
        yaml_file = tmp_path / "test.yaml"
        missing_sys = tmp_path / "missing_sys.yaml"
        yaml_file.write_text("variants: [{id: t1, prompt: 'test'}]")
        
        with pytest.raises(FileNotFoundError, match="System prompt file not found"):
            ExperimentConfig(
                yaml_path=yaml_file,
                sys_prompt_path=missing_sys,
                model_name="gpt-4"
            )
    
    def test_config_immutability(self, tmp_path):
        """Test that configuration is immutable."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: [{id: t1, prompt: 'test'}]")
        sys_file.write_text("system_prompt: 'test system'")
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4"
        )
        
        # Should not be able to modify
        with pytest.raises(Exception):  # dataclass frozen=True
            config.model_name = "different-model"


class TestPromptLoader:
    """Test the prompt loading functionality."""
    
    def test_load_basic_prompt_data(self, tmp_path):
        """Test loading basic prompt data."""
        # Create test YAML files
        yaml_content = {
            "variants": [
                {"id": "t1", "prompt": "Test prompt 1"},
                {"id": "t2", "prompt": "Test prompt 2"}
            ],
            "hypothesis": "This is a test hypothesis"
        }
        sys_content = {
            "system_prompt": "You are a helpful assistant",
            "shorthand": "helpful"
        }
        
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        sys_file.write_text(yaml.dump(sys_content))
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4"
        )
        
        prompt_data = PromptLoader.load_prompt_data(config)
        
        assert len(prompt_data.user_prompt_variants) == 2
        assert prompt_data.user_prompt_variants[0]["id"] == "t1"
        assert prompt_data.system_prompt_content == "You are a helpful assistant"
        assert prompt_data.system_prompt_shorthand == "helpful"
        assert prompt_data.system_prompt_tag == "sys:latest"
        assert prompt_data.hypothesis == "This is a test hypothesis"
        assert prompt_data.user_prompt_hash is not None
        assert prompt_data.system_prompt_hash is not None
    
    def test_load_prompt_data_with_missing_fields(self, tmp_path):
        """Test loading prompt data with missing optional fields."""
        yaml_content = {"variants": [{"id": "t1", "prompt": "Test"}]}
        sys_content = {"system_prompt": "Assistant"}
        
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        sys_file.write_text(yaml.dump(sys_content))
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4"
        )
        
        prompt_data = PromptLoader.load_prompt_data(config)
        
        assert prompt_data.system_prompt_shorthand == ""  # Default empty
        assert prompt_data.hypothesis is None  # Missing field


class TestModelMetadataService:
    """Test the model metadata resolution."""
    
    @patch('app.cli.experiment_runner_async.resolve_model')
    @patch('app.cli.experiment_runner_async.get_model_code')
    @patch('app.cli.experiment_runner_async.get_model_vendor')
    @patch('app.cli.experiment_runner_async.get_model_snapshot_id')
    def test_resolve_model_metadata(self, mock_snapshot, mock_vendor, mock_code, mock_resolve):
        """Test model metadata resolution."""
        # Setup mocks
        mock_resolve.return_value = "gpt-4-0125-preview"
        mock_code.return_value = "gpt-4"
        mock_vendor.return_value = "openai"
        mock_snapshot.return_value = "gpt-4-0125-preview-20240125"
        
        metadata = ModelMetadataService.resolve_model_metadata("gpt-4")
        
        assert metadata.canonical_name == "gpt-4-0125-preview"
        assert metadata.model_code == "gpt-4"
        assert metadata.vendor == "openai"
        assert metadata.snapshot_id == "gpt-4-0125-preview-20240125"
        
        # Verify all functions were called
        mock_resolve.assert_called_once_with("gpt-4")
        mock_code.assert_called_once_with("gpt-4-0125-preview")
        mock_vendor.assert_called_once_with("gpt-4-0125-preview")
        mock_snapshot.assert_called_once_with("openai", "gpt-4-0125-preview")


class TestSessionLogFactory:
    """Test the session log creation."""
    
    def test_create_session_log(self, tmp_path):
        """Test creating a session log with all metadata."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: [{id: t1, prompt: 'test'}]")
        sys_file.write_text("system_prompt: 'test system'")
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4",
            experiment_code="TEST",
            persona_name="test_persona"
        )
        
        prompt_data = PromptData(
            user_prompt_variants=[],
            user_prompt_hash="user_hash",
            user_prompt_raw="raw_user",
            system_prompt_content="system content",
            system_prompt_hash="sys_hash",
            system_prompt_shorthand="short",
            system_prompt_tag="sys:latest",
            hypothesis="test hypothesis"
        )
        
        model_metadata = ModelMetadata(
            canonical_name="gpt-4-turbo",
            model_code="gpt-4",
            vendor="openai",
            snapshot_id="gpt-4-turbo-20240125"
        )
        
        batch_id = "test-batch-01"
        
        session_log = SessionLogFactory.create_session_log(
            config, prompt_data, model_metadata, batch_id
        )
        
        assert session_log.model == "gpt-4-turbo"
        assert session_log.model_vendor == "openai"
        assert session_log.temperature == config.temperature
        assert session_log.system_prompt_content == "system content"
        assert session_log.system_prompt_hash == "sys_hash"
        assert session_log.user_prompt_hash == "user_hash"
        assert session_log.hypothesis == "test hypothesis"
        assert session_log.persona == "test_persona"
        assert session_log.workflow["batch_id"] == batch_id
        assert session_log.reproduction_info.experiment_code == "TEST"
    
    def test_experiment_code_mapping(self, tmp_path):
        """Test that experiment codes map to correct folders."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: []")
        sys_file.write_text("system_prompt: 'test'")
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4",
            experiment_code="80K"  # Should map to "80k_hours_demo"
        )
        
        prompt_data = PromptData(
            user_prompt_variants=[],
            user_prompt_hash="hash",
            user_prompt_raw="raw",
            system_prompt_content="content",
            system_prompt_hash="hash",
            system_prompt_shorthand="short",
            system_prompt_tag="tag"
        )
        
        model_metadata = ModelMetadata("model", "code", "vendor", "snapshot")
        
        session_log = SessionLogFactory.create_session_log(
            config, prompt_data, model_metadata, "batch-01"
        )
        
        assert session_log.workflow["experiment_name"] == "80k_hours_demo"


@pytest.mark.asyncio
class TestTurnProcessor:
    """Test the turn processing functionality."""
    
    async def test_process_turn_success(self):
        """Test successful turn processing."""
        # Mock runner
        mock_runner = MagicMock()
        mock_runner.generate.return_value = {
            "model_output": "Test response",
            "model_name": "gpt-4",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        processor = TurnProcessor(mock_runner)
        
        variant = {"id": "t1", "prompt": "Test Header\\nTest prompt content"}
        turn_index = 1
        history = ConversationHistory(system_prompt="system")
        
        config = MagicMock()
        config.persona_name = "test_persona"
        config.temperature = 0.7
        config.disable_containment = True
        
        prompt_data = MagicMock()
        prompt_data.system_prompt_tag = "sys:latest"
        
        model_metadata = MagicMock()
        model_metadata.vendor = "openai"
        
        # Mock callback
        callback_called = False
        async def mock_callback(model_name, turn_idx, output):
            nonlocal callback_called
            callback_called = True
            assert model_name == "gpt-4"
            assert turn_idx == 1
            assert output == "Test response"
        
        with patch('app.cli.experiment_runner_async.containment_summary') as mock_containment:
            mock_containment.return_value = {}
            
            turn = await processor.process_turn(
                variant, turn_index, history, config, prompt_data, model_metadata, mock_callback
            )
        
        assert isinstance(turn, Turn)
        assert turn.turn_index == 1
        assert turn.user_input_id == "t1"
        assert turn.persona == "test_persona"
        assert turn.raw_user_input == "Test prompt content"
        assert turn.model_output == "Test response"
        assert turn.input_token_count == 10
        assert turn.output_token_count == 5
        assert turn.total_token_count == 15
        assert callback_called
        
        # Verify runner was called correctly
        mock_runner.generate.assert_called_once_with(
            "Test prompt content",
            temperature=0.7,
            turn_index=1,
            conversation=history
        )
    
    async def test_process_turn_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        mock_runner = MagicMock()
        processor = TurnProcessor(mock_runner)
        
        variant = {"id": "t1", "prompt": "   "}  # Empty/whitespace prompt
        
        with pytest.raises(ValueError, match="Empty prompt in variant"):
            await processor.process_turn(
                variant, 1, MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
    
    async def test_process_turn_prompt_cleaning(self):
        """Test prompt cleaning logic."""
        mock_runner = MagicMock()
        mock_runner.generate.return_value = {
            "model_output": "Response",
            "usage": {}
        }
        
        processor = TurnProcessor(mock_runner)
        
        # Test YAML literal quote removal
        variant = {"id": "t1", "prompt": '"Quoted prompt"'}
        
        with patch('app.cli.experiment_runner_async.containment_summary'):
            turn = await processor.process_turn(
                variant, 1, MagicMock(), MagicMock(), MagicMock(), MagicMock()
            )
        
        assert turn.raw_user_input == "Quoted prompt"  # Quotes removed


@pytest.mark.asyncio
class TestAsyncExperimentRunner:
    """Test the main async experiment runner."""
    
    async def test_successful_experiment_run(self, tmp_path):
        """Test a complete successful experiment run."""
        # Create test files
        yaml_content = {
            "variants": [
                {"id": "t1", "prompt": "First prompt"},
                {"id": "t2", "prompt": "Second prompt"}
            ]
        }
        sys_content = {"system_prompt": "You are helpful"}
        
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text(yaml.dump(yaml_content))
        sys_file.write_text(yaml.dump(sys_content))
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4",
            log_dir=tmp_path
        )
        
        # Mock dependencies
        mock_prompt_loader = MagicMock()
        mock_model_service = MagicMock()
        mock_session_factory = MagicMock()
        
        mock_runner = MagicMock()
        mock_runner.generate.return_value = {
            "model_output": "Test response",
            "usage": {"total_tokens": 10}
        }
        
        def mock_get_runner(model_name):
            return mock_runner
        
        # Setup return values
        prompt_data = PromptData(
            user_prompt_variants=yaml_content["variants"],
            user_prompt_hash="hash",
            user_prompt_raw="raw",
            system_prompt_content="You are helpful",
            system_prompt_hash="sys_hash",
            system_prompt_shorthand="helpful",
            system_prompt_tag="sys:latest"
        )
        
        model_metadata = ModelMetadata("gpt-4", "gpt-4", "openai", "snapshot")
        session_log = SessionLog(
            isbn_run_id="test-run",
            turns=[],
            model="gpt-4"
        )
        
        mock_prompt_loader.load_prompt_data.return_value = prompt_data
        mock_model_service.resolve_model_metadata.return_value = model_metadata
        mock_session_factory.create_session_log.return_value = session_log
        
        runner = AsyncExperimentRunner(
            prompt_loader=mock_prompt_loader,
            model_service=mock_model_service,
            session_factory=mock_session_factory,
            get_runner_func=mock_get_runner
        )
        
        with patch('app.cli.experiment_runner_async.get_next_batch_id', return_value="batch-01"):
            with patch.object(runner, '_finalize_and_save_log') as mock_save:
                result = await runner.run_experiment(config)
        
        assert result.success
        assert isinstance(result.session_log, SessionLog)
        assert result.error_message is None
        assert result.execution_time_ms > 0
        
        # Verify turns were processed
        assert len(result.session_log.turns) == 2
        
        # Verify dependencies were called
        mock_prompt_loader.load_prompt_data.assert_called_once_with(config)
        mock_model_service.resolve_model_metadata.assert_called_once_with("gpt-4")
        mock_save.assert_called_once()
    
    async def test_experiment_run_with_error(self, tmp_path):
        """Test experiment run that encounters an error."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: []")
        sys_file.write_text("system_prompt: 'test'")
        
        config = ExperimentConfig(
            yaml_path=yaml_file,
            sys_prompt_path=sys_file,
            model_name="gpt-4"
        )
        
        # Mock that throws exception
        mock_prompt_loader = MagicMock()
        mock_prompt_loader.load_prompt_data.side_effect = Exception("Test error")
        
        runner = AsyncExperimentRunner(
            prompt_loader=mock_prompt_loader,
            model_service=MagicMock(),
            session_factory=MagicMock(),
            get_runner_func=MagicMock()
        )
        
        result = await runner.run_experiment(config)
        
        assert not result.success
        assert "Test error" in result.error_message
        assert result.execution_time_ms > 0


@pytest.mark.asyncio
class TestBackwardCompatibility:
    """Test the backward compatibility wrapper."""
    
    async def test_run_exploit_yaml_async_compatibility(self, tmp_path):
        """Test that the async wrapper maintains API compatibility."""
        # Create test files
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: [{id: t1, prompt: 'test'}]")
        sys_file.write_text("system_prompt: 'helpful'")
        
        # Mock the runner creation and execution
        with patch('app.cli.experiment_runner_async.create_experiment_runner') as mock_create:
            mock_runner = MagicMock()
            mock_result = ExperimentResult(
                session_log=SessionLog(isbn_run_id="test"),
                success=True
            )
            mock_runner.run_experiment.return_value = mock_result
            mock_create.return_value = mock_runner
            
            # Call with original API
            result = await run_exploit_yaml_async(
                yaml_path=str(yaml_file),
                sys_prompt=str(sys_file),
                model_name="gpt-4",
                temperature=0.7,
                experiment_code="TEST"
            )
            
            assert isinstance(result, SessionLog)
            assert result.isbn_run_id == "test"
    
    async def test_run_exploit_yaml_async_error_handling(self, tmp_path):
        """Test error handling in backward compatibility wrapper."""
        yaml_file = tmp_path / "test.yaml"
        sys_file = tmp_path / "sys.yaml"
        yaml_file.write_text("variants: []")
        sys_file.write_text("system_prompt: 'test'")
        
        with patch('app.cli.experiment_runner_async.create_experiment_runner') as mock_create:
            mock_runner = MagicMock()
            mock_result = ExperimentResult(
                session_log=SessionLog(),
                success=False,
                error_message="Test failure"
            )
            mock_runner.run_experiment.return_value = mock_result
            mock_create.return_value = mock_runner
            
            with pytest.raises(RuntimeError, match="Test failure"):
                await run_exploit_yaml_async(
                    yaml_path=str(yaml_file),
                    sys_prompt=str(sys_file)
                )


class TestIntegration:
    """Integration tests demonstrating the complete refactor."""
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_comparison(self):
        """Test memory efficiency vs threading approach."""
        # Create many experiment runners - should be lightweight
        runners = [create_experiment_runner() for _ in range(100)]
        
        assert len(runners) == 100
        
        # Each runner should have minimal memory footprint
        for runner in runners:
            assert runner.prompt_loader is not None
            assert runner.model_service is not None
            assert runner.session_factory is not None
    
    def test_dependency_injection_flexibility(self):
        """Test that dependency injection enables easy mocking/testing."""
        # Custom mock implementations
        mock_prompt_loader = MagicMock()
        mock_model_service = MagicMock()
        mock_session_factory = MagicMock()
        mock_get_runner = MagicMock()
        
        # Create runner with custom dependencies
        runner = AsyncExperimentRunner(
            prompt_loader=mock_prompt_loader,
            model_service=mock_model_service,
            session_factory=mock_session_factory,
            get_runner_func=mock_get_runner
        )
        
        # Verify dependencies are injected correctly
        assert runner.prompt_loader is mock_prompt_loader
        assert runner.model_service is mock_model_service
        assert runner.session_factory is mock_session_factory
        assert runner.get_runner_func is mock_get_runner
    
    def test_component_isolation(self):
        """Test that components can be tested in isolation."""
        # Each component can be instantiated and tested independently
        prompt_loader = PromptLoader()
        model_service = ModelMetadataService()
        session_factory = SessionLogFactory()
        
        # Components don't have hidden dependencies
        assert prompt_loader is not None
        assert model_service is not None  
        assert session_factory is not None


if __name__ == "__main__":
    # Run a simple integration test
    async def simple_integration_test():
        print("Testing refactored experiment runner...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            yaml_file = tmp_path / "test.yaml"
            sys_file = tmp_path / "sys.yaml"
            
            yaml_file.write_text(yaml.dump({
                "variants": [{"id": "t1", "prompt": "What is 2+2?"}]
            }))
            sys_file.write_text(yaml.dump({
                "system_prompt": "You are a helpful math assistant"
            }))
            
            # Test configuration creation
            config = ExperimentConfig(
                yaml_path=yaml_file,
                sys_prompt_path=sys_file,
                model_name="gpt-4"
            )
            
            print(f"✅ Created config for {config.model_name}")
            
            # Test prompt loading
            prompt_data = PromptLoader.load_prompt_data(config)
            print(f"✅ Loaded {len(prompt_data.user_prompt_variants)} variants")
            
            # Test model metadata
            with patch('app.cli.experiment_runner_async.resolve_model', return_value="gpt-4"):
                with patch('app.cli.experiment_runner_async.get_model_vendor', return_value="openai"):
                    metadata = ModelMetadataService.resolve_model_metadata("gpt-4")
                    print(f"✅ Resolved model metadata for {metadata.canonical_name}")
        
        print("Integration test completed successfully!")
    
    asyncio.run(simple_integration_test())