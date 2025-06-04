import types
from app.api_runners.anthropic_runner import AnthropicRunner
from app.core.context import ConversationHistory


def test_conversation_messages_claude3(monkeypatch):
    runner = AnthropicRunner(model_name="claude-3-opus")
    history = ConversationHistory(system_prompt="sys")
    history.append_user("hi")
    history.append_assistant("hello")
    history.append_user("how are you?")

    captured = {}

    class DummyResponse:
        def __init__(self):
            self.content = [types.SimpleNamespace(text="ok")]
            self.usage = types.SimpleNamespace(input_tokens=1, output_tokens=1)

    def fake_create(model, system, messages, temperature, top_p, max_tokens):
        captured["model"] = model
        captured["system"] = system
        captured["messages"] = messages
        return DummyResponse()

    monkeypatch.setattr(runner.client.messages, "create", fake_create)

    result = runner.generate("ignored", conversation=history)
    assert captured["messages"] == history.to_claude_format()["messages"]
    assert captured["system"] == history.system_prompt
    assert result["model_output"] == "ok"

def test_conversation_messages_legacy(monkeypatch):
    runner = AnthropicRunner(model_name="claude-2")
    history = ConversationHistory(system_prompt="sys")
    history.append_user("hi")
    history.append_assistant("hello")
    history.append_user("how are you?")

    captured = {}

    class DummyResponse:
        def __init__(self):
            self.completion = "ok"

    def fake_create(model, prompt, max_tokens_to_sample, temperature, top_p):
        captured["model"] = model
        captured["prompt"] = prompt
        return DummyResponse()

    monkeypatch.setattr(runner.client.completions, "create", fake_create)

    result = runner.generate("ignored", conversation=history)
    assert "hi" in captured["prompt"]
    assert "hello" in captured["prompt"]
    assert result["model_output"] == "ok"
