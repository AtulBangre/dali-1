import os
import re
from typing import Dict, List, Set, Any

class ProjectAnalyzer:
    """
    DALI-1 Deep Research Project Analyzer & Dependency Graph Crawler.
    Extracts multi-file architecture, imports, framework signatures, and dependencies.
    """
    def __init__(self, root_dir: str = "."):
        self.root_dir = os.path.abspath(root_dir)
        self.ignored_dirs = {
            "node_modules", ".git", "venv", ".venv", "__pycache__",
            ".next", "dist", "build", ".DS_Store"
        }
        self.code_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css",
            ".json", ".md", ".yml", ".yaml", "dockerfile"
        }

    def scan_directory_tree(self) -> Dict[str, Any]:
        """Crawls directory tree and gathers all project files."""
        tree = {"directories": [], "files": []}
        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [d for d in dirs if d not in self.ignored_dirs and not d.startswith(".")]
            rel_root = os.path.relpath(root, self.root_dir)
            if rel_root != ".":
                tree["directories"].append(rel_root)

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in self.code_extensions or f.lower() == "dockerfile":
                    rel_path = os.path.join(rel_root, f) if rel_root != "." else f
                    full_path = os.path.join(root, f)
                    try:
                        size = os.path.getsize(full_path)
                        tree["files"].append({
                            "path": rel_path,
                            "ext": ext,
                            "size": size
                        })
                    except OSError:
                        pass
        return tree

    def parse_file_dependencies(self, filepath: str) -> Dict[str, Any]:
        """Parses imports and dependencies in Python, JS, and TS files."""
        full_path = os.path.join(self.root_dir, filepath)
        if not os.path.exists(full_path):
            return {"imports": [], "lines": 0}

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception:
            return {"imports": [], "lines": 0}

        lines = content.splitlines()
        imports = set()

        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".py":
            for line in lines:
                line_str = line.strip()
                py_imp = re.match(r"^(?:from\s+([a-zA-Z0-9_.]+)\s+import|import\s+([a-zA-Z0-9_.]+))", line_str)
                if py_imp:
                    mod = py_imp.group(1) or py_imp.group(2)
                    imports.add(mod.split(".")[0])
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            for line in lines:
                line_str = line.strip()
                js_imp = re.search(r"(?:from\s+['\"]([^'\"]+)['\"]|require\(['\"]([^'\"]+)['\"]\))", line_str)
                if js_imp:
                    mod = js_imp.group(1) or js_imp.group(2)
                    imports.add(mod)

        return {
            "path": filepath,
            "lines": len(lines),
            "imports": sorted(list(imports))
        }

    def detect_tech_stacks(self, files: List[Dict[str, Any]]) -> List[str]:
        """Detects active frameworks and tools in the project."""
        stacks = set()
        file_paths = [f["path"].lower() for f in files]

        if any("requirements.txt" in p or "pyproject.toml" in p or p.endswith(".py") for p in file_paths):
            stacks.add("Python")
        if any("fastapi" in p or "main.py" in p for p in file_paths):
            stacks.add("FastAPI")
        if any(p.endswith((".tsx", ".jsx")) for p in file_paths):
            stacks.add("React")
        if any("next.config" in p or "app/" in p for p in file_paths):
            stacks.add("Next.js App Router")
        if any("tailwind" in p or "index.html" in p for p in file_paths):
            stacks.add("Tailwind CSS")
        if any("dockerfile" in p or "docker-compose" in p for p in file_paths):
            stacks.add("Docker")
        if any(p.endswith((".ts", ".tsx")) for p in file_paths):
            stacks.add("TypeScript")

        return sorted(list(stacks))

    def generate_project_cognitive_map(self) -> str:
        """Generates a structured cognitive architectural report for the AI agent."""
        tree = self.scan_directory_tree()
        files = tree["files"]
        stacks = self.detect_tech_stacks(files)

        report = [
            "# DALI-1 Deep Research Project Architecture Map",
            f"**Project Root:** `{self.root_dir}`",
            f"**Detected Tech Stacks:** {', '.join(stacks) if stacks else 'Generic Codebase'}",
            f"**Total Source Files:** {len(files)} files | **Directories:** {len(tree['directories'])} folders",
            "",
            "## 1. Directory Tree Overview"
        ]

        for d in sorted(tree["directories"])[:15]:
            report.append(f"- 📁 `{d}/`")
        if len(tree["directories"]) > 15:
            report.append(f"- ... and {len(tree['directories']) - 15} more directories.")

        report.append("\n## 2. Source Files & Module Dependencies")
        for f in files[:25]:
            dep = self.parse_file_dependencies(f["path"])
            imp_str = f" (Imports: {', '.join(dep['imports'][:4])})" if dep["imports"] else ""
            report.append(f"- 📄 `{f['path']}` [{dep['lines']} lines]{imp_str}")
        if len(files) > 25:
            report.append(f"- ... and {len(files) - 25} more files.")

        return "\n".join(report)

if __name__ == "__main__":
    analyzer = ProjectAnalyzer(".")
    print(analyzer.generate_project_cognitive_map())
