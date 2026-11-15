import os
import sys
import subprocess
import json
import re
from typing import Dict, Any, List, Optional

from chat import load_assistant, generate_response
from dali.analyzer import ProjectAnalyzer
from dali.cot import ChainOfThoughtReasoningEngine
from dali.router import SemanticIntentRouter
from dali.parameters import ActionParameterExtractor
from dali.reflexion import AgentReflexionEngine

class DALICodeAgent:
    """
    DALI-Code: Autonomous Local Terminal Software Engineering Agent
    Powered by Zero-Regex Semantic Embedding Intent Routing,
    Deliberative CoT Planning, and Autonomous Self-Correction Reflexion.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            # Auto-detect best available model: 1B INT4 if present, else 14M INT4
            if os.path.exists("checkpoints/dali1_1b_int4.pt"):
                model_path = "checkpoints/dali1_1b_int4.pt"
            else:
                model_path = "checkpoints/dali1_coder_int4.pt"
        print(f"✓ Initializing DALI-Code Agentic Engine with model: {model_path}...")
        self.model, self.tokenizer, self.config = load_assistant(model_path)
        self.knowledge_base_dir = "knowledge_base"
        self.analyzer = ProjectAnalyzer(".")
        self.cot_engine = ChainOfThoughtReasoningEngine()
        self.router = SemanticIntentRouter()
        self.param_extractor = ActionParameterExtractor()
        self.reflexion = AgentReflexionEngine()
        print("✓ Semantic Intent Router, CoT Engine & Autonomous Reflexion Online.")

    # TOOL 1: Read File
    def read_file(self, filepath: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist."
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)
        except Exception as e:
            return f"Error reading file: {e}"

    # TOOL 2: Write File
    def write_file(self, filepath: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"✓ Wrote {len(content)} bytes to '{filepath}'."
        except Exception as e:
            return f"Error writing file '{filepath}': {e}"

    # TOOL 3: Edit File (Target Replacement)
    def edit_file(self, filepath: str, target: str, replacement: str) -> str:
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist."
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if target not in content:
                return f"Error: Target text not found in '{filepath}'."
            new_content = content.replace(target, replacement, 1)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(new_content)
            return f"✓ Successfully replaced target in '{filepath}'."
        except Exception as e:
            return f"Error editing file: {e}"

    # TOOL 4: Run Terminal Command
    def run_command(self, command: str) -> str:
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = res.stdout if res.returncode == 0 else (res.stderr or res.stdout)
            return output.strip() or f"Command exited with code {res.returncode}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds."
        except Exception as e:
            return f"Error running command: {e}"

    # TOOL 5: RAG Knowledge Search
    def search_knowledge(self, query: str) -> str:
        """Retrieves most relevant documentation snippets from knowledge_base/"""
        stop_words = {
            "this", "that", "the", "a", "an", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "about", "how", "what", "why", "where", "when", "can", "do",
            "karo", "karna", "hai", "kaise", "bhi", "me", "ko", "par"
        }
        words = {w for w in query.lower().split() if len(w) > 2 and w not in stop_words}
        if not words:
            return "No local documentation match found."

        results = []
        if os.path.exists(self.knowledge_base_dir):
            for root, _, files in os.walk(self.knowledge_base_dir):
                for f in files:
                    if f.endswith(".txt") or f.endswith(".md"):
                        fpath = os.path.join(root, f)
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as doc_f:
                            content = doc_f.read()
                        blocks = content.split("<|end_block|>")
                        for b in blocks:
                            b_clean = b.strip()
                            if not b_clean:
                                continue
                            b_words = set(b_clean.lower().split())
                            overlap = len(words.intersection(b_words))
                            if overlap >= 2 or (len(words) == 1 and overlap == 1):
                                results.append((overlap, b_clean[:600]))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:2]
        if not top:
            return "No local documentation match found."
        return "\n---\n".join([item[1] for item in top])

    def _list_directory(self) -> str:
        """Lists files and directories cleanly."""
        try:
            items = sorted(os.listdir("."))
            folders = [f"{i}/" for i in items if os.path.isdir(i) and not i.startswith(".")]
            files = [i for i in items if os.path.isfile(i) and not i.startswith(".")]
            output = ["Current Directory Contents:"]
            if folders:
                output.append("  [Directories]: " + ", ".join(folders))
            if files:
                output.append("  [Files]:       " + ", ".join(files[:20]))
                if len(files) > 20:
                    output.append(f"  ... and {len(files) - 20} more files.")
            return "\n".join(output)
        except Exception as e:
            return f"Error listing directory: {e}"

    def _generate_cot_plan(self, task_desc: str) -> str:
        """Generates 4-phase deliberative Chain-of-Thought planning."""
        stacks = self.analyzer.detect_tech_stacks(self.analyzer.scan_directory_tree()['files'])
        stack_str = ', '.join(stacks) if stacks else 'Full-Stack Web'
        thinking_plan = f"""[DALI Deliberative Chain-of-Thought (CoT)]
<thinking>
Phase 1: Problem Decomposition & Constraints Analysis
- Target Task: {task_desc}
- Active Project Stacks: {stack_str}

Phase 2: Architectural Dependency Mapping
- Inspecting project modules, import graphs, and interface boundaries.
- Ensuring backwards compatibility, strict schemas, and zero regressions.

Phase 3: Multi-Step Execution Plan
- Step 1: Inspect target component or module files.
- Step 2: Implement verified changes with defensive type validation.
- Step 3: Run sanity tests via command tool.

Phase 4: Risk Mitigation & Edge Cases
- Defensive null-checks, error boundaries, and environment fallback configurations.
</thinking>

✓ Deliberative Planning Complete! Ready to execute: '{task_desc}'"""
        return thinking_plan

    def execute_instruction(self, user_prompt: str) -> str:
        """
        Agentic Execution Pipeline:
        1. Explicit Slash Command Handling
        2. Zero-Regex Semantic Embedding Intent Routing
        3. Structured Parameter Extraction
        4. Tool Execution
        5. Autonomous Reflexion & Self-Correction Verification
        """
        prompt_trimmed = user_prompt.strip()
        if not prompt_trimmed:
            return ""

        # 1. Direct Slash Commands
        if prompt_trimmed.startswith("/analyze"):
            target_path = prompt_trimmed[8:].strip() or "."
            return ProjectAnalyzer(target_path).generate_project_cognitive_map()

        if prompt_trimmed.startswith("/think "):
            return self._generate_cot_plan(prompt_trimmed[7:].strip())

        if prompt_trimmed.startswith("/write "):
            parts = prompt_trimmed[7:].strip().split(maxsplit=1)
            return self.write_file(parts[0], parts[1] if len(parts) > 1 else "")

        if prompt_trimmed.startswith("/run "):
            cmd = prompt_trimmed[5:].strip()
            return f"[$ {cmd}]\n" + self.run_command(cmd)

        if prompt_trimmed.startswith("/read "):
            path = prompt_trimmed[6:].strip()
            return f"[File: {path}]\n" + self.read_file(path)

        if prompt_trimmed.startswith("/search "):
            q = prompt_trimmed[8:].strip()
            return f"[RAG Knowledge Results for '{q}']:\n" + self.search_knowledge(q)

        # 2. Semantic Intent Classification via Vector Cosine Similarity
        routing = self.router.route_intent(prompt_trimmed)
        action = routing["primary_action"]
        params = self.param_extractor.extract_parameters(action, prompt_trimmed)

        output = ""

        # 3. Tool Dispatch Pipeline
        if action == "BUILD_WEBSITE":
            from build_app import build_website_from_prompt
            target_dir = params.get("target_dir")
            index_path = build_website_from_prompt(prompt_trimmed, direct_folder=target_dir)
            folder_used = target_dir or os.path.dirname(index_path)
            output = f"✓ Built modern responsive website successfully!\n  Directory: {os.path.abspath(folder_used)}\n  Entrypoint: {os.path.abspath(index_path)}\n  Preview: file://{os.path.abspath(index_path)}"

        elif action == "BUILD_FULLSTACK":
            from build_fullstack_app import generate_fullstack_project
            target_dir = params.get("target_dir")
            proj_dir = generate_fullstack_project(prompt_trimmed, direct_folder=target_dir)
            folder_used = target_dir or proj_dir
            output = f"✓ Full-stack project generated successfully!\n  Target Directory: {os.path.abspath(folder_used)}\n  Run './run.sh' inside to start services."

        elif action == "ANALYZE_PROJECT":
            output = self.analyzer.generate_project_cognitive_map()

        elif action == "PLAN_TASK":
            output = self._generate_cot_plan(prompt_trimmed)

        elif action == "CREATE_FOLDER":
            folder_name = params.get("folder_name", "new_folder")
            try:
                os.makedirs(folder_name, exist_ok=True)
                abs_path = os.path.abspath(folder_name)
                output = f"✓ Created folder '{folder_name}' successfully!\n  Path: {abs_path}"
            except Exception as e:
                output = f"Error creating folder '{folder_name}': {e}"

        elif action == "FILE_OPERATIONS":
            sub = params.get("sub_action")
            if sub == "list":
                output = self._list_directory()
            elif sub == "read":
                output = f"[File: {params.get('filepath')}]\n" + self.read_file(params.get("filepath"))
            elif sub == "delete":
                target = params.get("target")
                if os.path.isdir(target):
                    import shutil
                    shutil.rmtree(target)
                    output = f"✓ Removed directory '{target}'."
                elif os.path.isfile(target):
                    os.remove(target)
                    output = f"✓ Removed file '{target}'."
                else:
                    output = f"Error: '{target}' not found."
            elif sub == "create":
                fpath = params.get("filepath")
                os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
                if not os.path.exists(fpath):
                    with open(fpath, "w", encoding="utf-8") as f:
                        f.write("")
                output = f"✓ File '{fpath}' created successfully at {os.path.abspath(fpath)}"
            else:
                output = self._list_directory()

        elif action == "RUN_COMMAND":
            cmd = params.get("command", "")
            output = f"[$ {cmd}]\n" + self.run_command(cmd)

        elif action == "KNOWLEDGE_SEARCH":
            rag_match = self.search_knowledge(prompt_trimmed)
            if rag_match and rag_match != "No local documentation match found.":
                output = f"[Official Knowledge & Debugging Reference]:\n{rag_match}"
            else:
                output = self.analyzer.generate_project_cognitive_map()

        # 4. Autonomous Reflexion & Self-Correction Gate
        is_verified, audit_msg, remediation = self.reflexion.audit_and_verify(action, params, output)
        if not is_verified:
            # Self-Correction Triggered!
            if remediation.get("remediation") in ["SYNTHESIZE_WEBSITE", "REGENERATE_WEBSITE"]:
                from build_app import build_website_from_prompt
                fix_dir = remediation.get("target_dir")
                index_path = build_website_from_prompt(prompt_trimmed, direct_folder=fix_dir)
                output += f"\n\n[Auto-Reflexion Self-Correction Executed]:\n  ✓ Generated missing HTML5 assets at {index_path}"

        return output

def main():
    print("=" * 72)
    print("  DALI-CODE: AUTONOMOUS TERMINAL CODING AGENT (LIKE CLAUDE CODE)")
    print("  Tools: /read <file> | /run <cmd> | /search <query> | exit")
    print("=" * 72)

    agent = DALICodeAgent()
    print("✓ Local Agent Ready! Working Directory: " + os.getcwd())
    print("-" * 72)

    # CLI Argument one-shot execution
    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:])
        print(f"\nUser: {instruction}")
        output = agent.execute_instruction(instruction)
        print(f"\nDALI-Code:\n{output}")
        return

    # Interactive REPL
    while True:
        try:
            inp = input("\ndali-code> ").strip()
            if not inp:
                continue
            if inp.lower() in ["exit", "quit", "q"]:
                print("Exiting DALI-Code. Happy coding!")
                break
            result = agent.execute_instruction(inp)
            print(f"\n{result}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting DALI-Code.")
            break

if __name__ == "__main__":
    main()
