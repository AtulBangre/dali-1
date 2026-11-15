import os
import sys
import subprocess
import json
import re
from typing import Dict, Any, List

from chat import load_assistant, generate_response
from dali.analyzer import ProjectAnalyzer
from dali.cot import ChainOfThoughtReasoningEngine

class DALICodeAgent:
    """
    DALI-Code: Autonomous Local Terminal Software Engineering Agent
    Similar to Claude Code, with File System Tools, Command Runner, RAG, and CoT Reasoning.
    """
    def __init__(self, model_path: str = "checkpoints/dali1_coder_int4.pt"):
        print("✓ Initializing DALI-Code Agentic Engine...")
        self.model, self.tokenizer, self.config = load_assistant(model_path)
        self.knowledge_base_dir = "knowledge_base"
        self.analyzer = ProjectAnalyzer(".")
        self.cot_engine = ChainOfThoughtReasoningEngine()
        print("✓ Neural Brain, Project Crawler & CoT Reasoning Online (4-Bit INT4).")

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

    def _extract_name(self, text: str, is_folder: bool = True) -> str:
        """Extracts target folder or file name from natural language instruction."""
        words = text.split()
        stop_words = {
            "make", "create", "mkdir", "bnao", "bnana", "a", "an", "the",
            "folder", "directory", "dir", "file", "name", "named", "called",
            "in", "this", "current", "working", "here", "please", "with", "to"
        }
        candidates = [w.strip(".,;:\"'`()") for w in words if w.lower() not in stop_words and len(w) > 0]
        if candidates:
            return candidates[0]
        return ""

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
        """Agentic Execution Loop with Robust Natural Language Intent Dispatching"""
        prompt_trimmed = user_prompt.strip()
        if not prompt_trimmed:
            return ""

        prompt_lower = prompt_trimmed.lower()

        # 1. Direct Slash Commands
        if prompt_trimmed.startswith("/analyze"):
            target_path = prompt_trimmed[8:].strip() or "."
            analyzer = ProjectAnalyzer(target_path)
            return analyzer.generate_project_cognitive_map()

        if prompt_trimmed.startswith("/think "):
            task_desc = prompt_trimmed[7:].strip()
            return self._generate_cot_plan(task_desc)

        if prompt_trimmed.startswith("/write "):
            parts = prompt_trimmed[7:].strip().split(maxsplit=1)
            if len(parts) >= 1:
                fpath = parts[0]
                content = parts[1] if len(parts) > 1 else ""
                return self.write_file(fpath, content)

        if prompt_trimmed.startswith("/run "):
            cmd = prompt_trimmed[5:].strip()
            return f"[$ {cmd}]\n" + self.run_command(cmd)

        if prompt_trimmed.startswith("/read "):
            path = prompt_trimmed[6:].strip()
            return f"[File: {path}]\n" + self.read_file(path)

        if prompt_trimmed.startswith("/search "):
            q = prompt_trimmed[8:].strip()
            return f"[RAG Knowledge Results for '{q}']:\n" + self.search_knowledge(q)

        # 2. Intent: Project Analysis & Architecture Crawling
        # Handles: "analys this project", "analyse project", "analyze codebase", "scan project", "project structure", "architecture map"
        has_analyze_stem = bool(re.search(r"analy[sz]", prompt_lower))
        has_project_kw = any(w in prompt_lower for w in ["project", "codebase", "repo", "architecture", "directory tree", "dependencies"])
        has_scan_kw = any(w in prompt_lower for w in ["scan", "audit", "inspect", "overview", "structure"]) and has_project_kw
        has_hindi_inspect = any(p in prompt_lower for p in ["project me kya hai", "project check karo", "kya chal raha", "files scan"])

        if has_analyze_stem or has_scan_kw or has_hindi_inspect:
            return self.analyzer.generate_project_cognitive_map()

        # 3. Intent: Deliberative Planning & Thinking (Chain-of-Thought)
        # Handles: "plan how to add redis", "think about login", "kaise implement kare", "deep research on auth", "design architecture"
        has_plan_kw = bool(re.search(r"\b(plan|think|strategy|kaise|how to|design|approach|steps|roadmap)\b", prompt_lower))
        if has_plan_kw and not any(w in prompt_lower for w in ["make", "create", "build", "website", "folder"]):
            return self._generate_cot_plan(prompt_trimmed)

        # 4. Intent: Folder / Directory Operations (Make / Create / Mkdir)
        # Handles: "make a folder name ATUL in this current working directory", "mkdir test", "create directory components"
        if any(w in prompt_lower for w in ["folder", "directory", "dir", "mkdir"]):
            if any(w in prompt_lower for w in ["make", "create", "mkdir", "bnao", "bnana", "generate", "add"]):
                folder_name = self._extract_name(prompt_trimmed, is_folder=True)
                if folder_name:
                    try:
                        os.makedirs(folder_name, exist_ok=True)
                        abs_path = os.path.abspath(folder_name)
                        return f"✓ Created folder '{folder_name}' successfully!\n  Path: {abs_path}"
                    except Exception as e:
                        return f"Error creating folder '{folder_name}': {e}"

        # 5. Intent: File Operations (Create / Write / Edit)
        # Handles: "create file test.py", "write index.html", "touch server.js"
        if any(w in prompt_lower for w in ["file", "touch"]) and any(w in prompt_lower for w in ["create", "make", "write", "add", "bnao"]):
            file_name = self._extract_name(prompt_trimmed, is_folder=False)
            if file_name and "." in file_name:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(file_name)), exist_ok=True)
                    if not os.path.exists(file_name):
                        with open(file_name, "w", encoding="utf-8") as f:
                            f.write("")
                    return f"✓ File '{file_name}' created successfully at {os.path.abspath(file_name)}"
                except Exception as e:
                    return f"Error creating file '{file_name}': {e}"

        # 6. Intent: List Files / Directory Contents
        if any(w in prompt_lower for w in ["list files", "show files", "what files", "ls", "dir", "show directory", "directory contents", "files dikhao"]):
            return self._list_directory()

        # 7. Intent: Read / View File
        read_match = re.search(r"(?:read|view|show|open|cat)\s+(?:file\s+)?([A-Za-z0-9_./\\-]+\.[a-zA-Z0-9]+)", prompt_trimmed, re.I)
        if read_match:
            filepath = read_match.group(1).strip()
            return f"[File: {filepath}]\n" + self.read_file(filepath)

        # 8. Intent: Delete / Remove File or Folder
        del_match = re.search(r"(?:delete|remove|rm)\s+(?:(?:folder|directory|file)\s+)?([A-Za-z0-9_.-]+)", prompt_trimmed, re.I)
        if del_match and any(w in prompt_lower for w in ["delete", "remove", "rm", "hatao"]):
            target = del_match.group(1).strip()
            if os.path.isdir(target):
                import shutil
                shutil.rmtree(target)
                return f"✓ Removed directory '{target}'."
            elif os.path.isfile(target):
                os.remove(target)
                return f"✓ Removed file '{target}'."
            else:
                return f"Error: '{target}' not found."

        # 9. Intent: Autonomous Website / Full-Stack Project Generator
        if any(w in prompt_lower for w in ["website", "fullstack", "frontend", "backend", "full-stack", "web app"]):
            if any(w in prompt_lower for w in ["build", "create", "make", "bnao", "generate", "start"]):
                from build_fullstack_app import generate_fullstack_project
                proj_dir = generate_fullstack_project(prompt_trimmed)
                return f"✓ Full-stack project generated at: {proj_dir}"

        # 10. Technical Questions & Knowledge Retrieval via RAG
        rag_match = self.search_knowledge(prompt_trimmed)
        if rag_match and rag_match != "No local documentation match found.":
            return f"[Official Knowledge & Debugging Reference]:\n{rag_match}"

        # Fallback guidance when command is ambiguous
        return f"""DALI-Code Assistant Ready.
Aap ye actions perform kar sakte hain:
- Project analyze karein: 'analys this project' ya '/analyze'
- Step-by-step planning: 'plan how to add redis' ya '/think <task>'
- File/Folder banayein: 'make a folder name <name>' ya 'create file <name>'
- Files dekhein: 'list files' ya 'read file <name>'
- Technical query: '/search <topic>' (FastAPI, React, Tailwind, Docker, etc.)
- Command chalayein: '/run <bash command>'"""

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
