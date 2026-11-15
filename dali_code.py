import os
import sys
import subprocess
import json
import re
from typing import Dict, Any, List

from chat import load_assistant, generate_response

class DALICodeAgent:
    """
    DALI-Code: Autonomous Local Terminal Software Engineering Agent
    Similar to Claude Code, with File System Tools, Command Runner, and RAG.
    """
    def __init__(self, model_path: str = "checkpoints/dali1_coder_int4.pt"):
        print("✓ Initializing DALI-Code Agentic Engine...")
        self.model, self.tokenizer, self.config = load_assistant(model_path)
        self.knowledge_base_dir = "knowledge_base"
        print("✓ Neural Brain & Bounded Memory Bank Online (4-Bit INT4).")

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
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully written {len(content)} bytes to '{filepath}'."
        except Exception as e:
            return f"Error writing file: {e}"

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
            return f"Successfully replaced target in '{filepath}'."
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
        results = []
        words = set(query.lower().split())

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
                            if overlap > 0:
                                results.append((overlap, b_clean[:500]))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:3]
        if not top:
            return "No local documentation match found."
        return "\n---\n".join([item[1] for item in top])

    def execute_instruction(self, user_prompt: str) -> str:
        """Agentic Execution Loop with Natural Language Intent Dispatching"""
        prompt_trimmed = user_prompt.strip()
        prompt_lower = prompt_trimmed.lower()

        # 1. Direct Slash Commands
        if prompt_trimmed.startswith("/run "):
            cmd = prompt_trimmed[5:].strip()
            return f"[$ {cmd}]\n" + self.run_command(cmd)

        if prompt_trimmed.startswith("/read "):
            path = prompt_trimmed[6:].strip()
            return f"[File: {path}]\n" + self.read_file(path)

        if prompt_trimmed.startswith("/search "):
            q = prompt_trimmed[8:].strip()
            return f"[RAG Knowledge Results for '{q}']:\n" + self.search_knowledge(q)

        # 2. Natural Language Intent: Create / Make Folder / Directory
        folder_match = re.search(r"(?:make|create|mkdir)\s+(?:a\s+)?(?:folder|directory)?\s*(?:named?\s+|called\s+)?([A-Za-z0-9_.-]+)", prompt_trimmed, re.I)
        if folder_match and any(w in prompt_lower for w in ["folder", "directory", "mkdir"]):
            folder_name = folder_match.group(1).strip()
            if folder_name.lower() in ["folder", "directory", "a", "the"]:
                # Try finding token after 'named' or 'called'
                parts = prompt_trimmed.split()
                if "name" in parts:
                    folder_name = parts[parts.index("name") + 1]
                elif "named" in parts:
                    folder_name = parts[parts.index("named") + 1]
            try:
                os.makedirs(folder_name, exist_ok=True)
                abs_path = os.path.abspath(folder_name)
                return f"✓ Created folder '{folder_name}' successfully!\n  Path: {abs_path}"
            except Exception as e:
                return f"Error creating folder '{folder_name}': {e}"

        # 3. Natural Language Intent: List / Show Files
        if any(phrase in prompt_lower for phrase in ["list files", "show files", "what files", "ls", "dir", "show directory"]):
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

        # 4. Natural Language Intent: Read / View File
        read_match = re.search(r"(?:read|view|show|open|cat)\s+(?:file\s+)?([A-Za-z0-9_./\\-]+\.[a-zA-Z0-9]+)", prompt_trimmed, re.I)
        if read_match:
            filepath = read_match.group(1).strip()
            return f"[File: {filepath}]\n" + self.read_file(filepath)

        # 5. Natural Language Intent: Delete File / Folder
        del_match = re.search(r"(?:delete|remove|rm)\s+(?:(?:folder|directory|file)\s+)?([A-Za-z0-9_.-]+)", prompt_trimmed, re.I)
        if del_match and any(w in prompt_lower for w in ["delete", "remove", "rm"]):
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

        # 6. Natural Language Intent: Build Website / Project
        if any(w in prompt_lower for w in ["build website", "create website", "make website", "build fullstack", "create app"]):
            from build_fullstack_app import generate_fullstack_project
            proj_dir = generate_fullstack_project(prompt_trimmed)
            return f"✓ Full-stack project generated at: {proj_dir}"

        # 7. Code Knowledge & Technical Question: Augmented by RAG
        rag_context = self.search_knowledge(prompt_trimmed)
        prompt_formatted = f"<|user|>\n{prompt_trimmed}\n<|assistant|>"

        response = generate_response(self.model, self.tokenizer, prompt_formatted, max_new_tokens=120, temperature=0.2, top_k=2)
        return response

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
