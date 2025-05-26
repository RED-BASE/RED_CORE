from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional

@dataclass
class MessageTurn:
    role: Literal["user", "assistant"]
    content: str

@dataclass
class ConversationContext:
    rendered_prompt: str
    user_input: str = ""
    model_output: str = ""
    persona: str = "none"
    system_prompt_tag: str = "standard_guardrails:latest"
    meta: Dict[str, str] = field(default_factory=dict)

    def update_output(self, text: str) -> None:
        self.model_output = text

    def as_dict(self) -> Dict:
        return {
            "rendered_prompt": self.rendered_prompt,
            "user_input": self.user_input,
            "model_output": self.model_output,
            "persona": self.persona,
            "system_prompt_tag": self.system_prompt_tag,
            "meta": self.meta,
        }

@dataclass
class ConversationHistory:
    system_prompt: Optional[str] = None
    turns: List[MessageTurn] = field(default_factory=list)

    def append_user(self, content: str):
        self.turns.append(MessageTurn(role="user", content=content))

    def append_assistant(self, content: str):
        self.turns.append(MessageTurn(role="assistant", content=content))

    def to_openai_format(self) -> List[Dict[str, str]]:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages += [t.__dict__ for t in self.turns]
        return messages

    def to_claude_format(self) -> Dict:
        return {
            "system": self.system_prompt or "",
            "messages": [t.__dict__ for t in self.turns]
        }

    def to_gemini_format(self) -> List[Dict[str, Dict[str, str]]]:
        messages = []
        for idx, turn in enumerate(self.turns):
            if turn.role not in {"user", "assistant"}:
                continue

            content = turn.content.strip()
            if idx == 0 and self.system_prompt:
                content = f"{self.system_prompt.strip()}\n\n{content}"

            if content:
                messages.append({
                    "role": "user" if turn.role == "user" else "model",
                    "parts": [{"text": content}],
                })

        return messages
