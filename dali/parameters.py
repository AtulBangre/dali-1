import os
import re
from typing import Dict, Any, Optional

class ActionParameterExtractor:
    """
    Standardized Parameter Extractor for DALI-1 Agentic Tools.
    Converts unstructured natural language into clean, strongly-typed JSON parameter schemas.
    """
    def __init__(self):
        pass

    def extract_parameters(self, action: str, prompt: str) -> Dict[str, Any]:
        """Extracts tool-specific parameters based on identified action."""
        if action in ["BUILD_WEBSITE", "BUILD_FULLSTACK"]:
            return self._extract_web_parameters(prompt)
        elif action == "CREATE_FOLDER":
            return self._extract_folder_parameters(prompt)
        elif action == "FILE_OPERATIONS":
            return self._extract_file_parameters(prompt)
        elif action == "RUN_COMMAND":
            return self._extract_command_parameters(prompt)
        elif action == "PLAN_TASK":
            return {"task": prompt.strip()}
        elif action == "ANALYZE_PROJECT":
            return {"target_path": "."}
        elif action == "KNOWLEDGE_SEARCH":
            return {"query": prompt.strip()}
        return {}

    def _extract_web_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extracts destination folder and topic for website/app generator."""
        # Find explicit target folder (e.g. save in ATUL folder, inside portfolio, etc.)
        target_dir = None
        m = re.search(
            r'(?:save\s+(?:all\s+)?(?:the\s+)?files?\s+(?:in|inside|to)|inside|into|in)\s+(?:this\s+)?(?:folder\s+|directory\s+)?([A-Za-z0-9_.-]+)(?:\s+folder|\s+directory)?',
            prompt, re.I
        )
        if m:
            candidate = m.group(1).strip()
            if candidate.lower() not in ["this", "a", "an", "the", "folder", "directory", "current", "working"]:
                target_dir = candidate

        # Clean prompt topic
        topic = prompt
        # Theme detection
        theme = "dark" if any(w in prompt.lower() for w in ["dark", "black", "night", "cyber"]) else "modern_light"

        return {
            "topic": topic,
            "target_dir": target_dir,
            "theme": theme
        }

    def _extract_folder_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extracts pure folder name to create."""
        words = prompt.split()
        stop_words = {
            "make", "create", "mkdir", "bnao", "bnana", "a", "an", "the",
            "folder", "directory", "dir", "name", "named", "called",
            "in", "this", "current", "working", "here", "please", "with", "to"
        }
        candidates = [w.strip(".,;:\"'`()") for w in words if w.lower() not in stop_words and len(w) > 0]
        folder_name = candidates[0] if candidates else "new_folder"
        return {"folder_name": folder_name}

    def _extract_file_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extracts sub-action and file paths."""
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["list", "show files", "ls", "dir", "dikhao"]):
            return {"sub_action": "list", "target": "."}
        
        # Read file
        read_m = re.search(r"(?:read|view|show|open|cat)\s+(?:file\s+)?([A-Za-z0-9_./\\-]+\.[a-zA-Z0-9]+)", prompt, re.I)
        if read_m:
            return {"sub_action": "read", "filepath": read_m.group(1).strip()}

        # Delete file
        del_m = re.search(r"(?:delete|remove|rm)\s+(?:(?:folder|file)\s+)?([A-Za-z0-9_.-]+)", prompt, re.I)
        if del_m and any(w in p_lower for w in ["delete", "remove", "rm", "hatao"]):
            return {"sub_action": "delete", "target": del_m.group(1).strip()}

        # Create/write file
        create_m = re.search(r"(?:create|make|write|touch)\s+(?:file\s+)?([A-Za-z0-9_.-]+\.[a-zA-Z0-9]+)", prompt, re.I)
        if create_m:
            return {"sub_action": "create", "filepath": create_m.group(1).strip()}

        return {"sub_action": "unknown"}

    def _extract_command_parameters(self, prompt: str) -> Dict[str, Any]:
        clean = re.sub(r"^/(?:run|exec)\s+", "", prompt).strip()
        return {"command": clean}
