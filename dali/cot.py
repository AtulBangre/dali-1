import re
from typing import Dict, Any, List

class ChainOfThoughtReasoningEngine:
    """
    DALI-1 Deliberative Chain-of-Thought (CoT) Reasoning Engine.
    Structures deep thinking into 4 phases before generating code or actions:
    1. Problem Understanding & Constraints
    2. Architecture & Dependency Mapping
    3. Step-by-Step Execution Plan
    4. Verification & Edge Case Defense
    """
    def __init__(self):
        pass

    def formulate_reasoning_prompt(self, user_task: str, project_context: str = "") -> str:
        """Enforces structured deliberative thinking format."""
        prompt = f"""<|system|>
You are DALI-Reasoning Engine. For any software task, you MUST reason step-by-step inside <thinking> blocks before producing code.
<|user|>
Task: {user_task}

Project Context:
{project_context[:400] if project_context else "Single-component or standalone task"}

Formulate your thinking and solution.
<|assistant|>
<thinking>
1. Problem Understanding & Constraints:
- Target goal: {user_task}
- Architecture requirements: Robust, modular, production-ready.

2. Architecture & Dependency Analysis:
- Key components involved: Layout, state management, API contract.
- Edge cases: Error handling, responsive constraints, null-checks.

3. Step-by-Step Execution Plan:
- Step 1: Initialize required data structures and dependencies.
- Step 2: Implement core logic with strict typing and defensive checks.
- Step 3: Validate outputs against edge cases.
</thinking>

### Implementation Plan & Solution:
"""
        return prompt

    def parse_reasoning_output(self, raw_output: str) -> Dict[str, str]:
        """Separates thinking traces from the final executable solution."""
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", raw_output, re.DOTALL)
        thinking_trace = thinking_match.group(1).strip() if thinking_match else "Direct computation executed."
        
        # Clean solution without thinking tags
        solution = re.sub(r"<thinking>.*?</thinking>", "", raw_output, flags=re.DOTALL).strip()
        
        return {
            "thinking": thinking_trace,
            "solution": solution
        }

if __name__ == "__main__":
    engine = ChainOfThoughtReasoningEngine()
    test_prompt = engine.formulate_reasoning_prompt("Add dark mode toggle to navigation bar", "Tailwind CSS + React 19")
    print(test_prompt[:400] + "...")
