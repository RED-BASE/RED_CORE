import types
from app.api_runners.openai_runner import OpenAIRunner
from app.core.context import ConversationHistory

def test_conversation_messages(monkeypatch):
    runner = OpenAIRunner(api_key="test", model_name="gpt-4o")
    history = ConversationHistory(system_prompt="sys")
    history.append_user("hi")
    history.append_assistant("hello")
    history.append_user("how are you?")

    captured = {}

    class DummyResponse:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            self.usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)

    def fake_create(model, messages, **kw):
        captured['model'] = model
        captured['messages'] = messages
        return DummyResponse()

    monkeypatch.setattr(runner.client.chat.completions, "create", fake_create)

    result = runner.generate("how are you?", conversation=history)
    assert captured['messages'] == history.to_openai_format()
    assert result['model_output'] == "ok"
