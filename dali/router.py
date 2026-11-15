import math
import re
from typing import Dict, List, Tuple, Any

class SemanticIntentRouter:
    """
    DALI-1 Zero-Regex Semantic Embedding Intent Router.
    Uses subword character n-gram TF-IDF vectorization and cosine similarity
    to classify user prompts into agentic actions without hardcoded string matching.
    Robust against typos (e.g. 'analys'), Hinglish ('bnao'), and complex phrasing.
    """
    def __init__(self):
        # 1. Action Definitions with Multi-Lingual & Technical Semantic Anchors
        self.action_anchors: Dict[str, List[str]] = {
            "BUILD_WEBSITE": [
                "make a portfolio website for developer save files in folder",
                "create a modern responsive website landing page web app",
                "build frontend website using html5 tailwind css javascript",
                "generate portfolio site inside directory",
                "design web page responsive user interface",
                "ek modern website bnao portfolio web developer ke liye",
                "landing page create karo frontend web application",
                "build modern ui website with dark mode and interactive sections"
            ],
            "BUILD_FULLSTACK": [
                "build fullstack project with frontend and backend",
                "create full stack web app with fastapi react and database",
                "generate fullstack application with docker-compose rest api",
                "backend aur frontend alag alag folder me complete project bnao",
                "develop fullstack web platform with python api and responsive ui"
            ],
            "ANALYZE_PROJECT": [
                "analyze this project architecture and dependency map",
                "analys this project scan codebase dependencies",
                "analyse repository inspect directory tree and imports",
                "project structure scan karo codebase me kya hai",
                "audit codebase show file tree and detected tech stacks",
                "inspect project overview and source file dependencies"
            ],
            "PLAN_TASK": [
                "plan how to implement feature step by step",
                "think about architecture design deliberative chain of thought",
                "how to add redis caching layer to fastapi",
                "kaise implement kare step by step plan strategy roadmap",
                "deep research and architectural reasoning before writing code"
            ],
            "CREATE_FOLDER": [
                "make a folder named ATUL in current working directory",
                "mkdir directory create empty folder",
                "ek naya folder bnao directory add karo",
                "create folder for backups",
                "make new directory named components"
            ],
            "FILE_OPERATIONS": [
                "create file test.py write code inside file",
                "read file view file contents cat script.js",
                "write file index.html update existing code",
                "list files show directory contents ls dir",
                "delete file remove folder rm -rf cleanup",
                "file dikhao kya files hai current directory me"
            ],
            "RUN_COMMAND": [
                "run bash command in terminal execute shell",
                "git status git commit push to origin",
                "exec command npm test pytest run script"
            ],
            "KNOWLEDGE_SEARCH": [
                "explain react 19 server actions nextjs app router",
                "fastapi database connection pool operational error fix",
                "tailwind css grid flexbox utility classes official documentation",
                "docker multi stage build caching best practices"
            ]
        }

        # Build vocabulary of character 3-grams and 4-grams for typo resilience
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._build_vector_space()

    def _extract_ngrams(self, text: str) -> List[str]:
        """Extracts word tokens and character 3-grams & 4-grams."""
        clean = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        words = clean.split()
        features = list(words)

        for w in words:
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    features.append(w[i:i+3])
            if len(w) >= 4:
                for i in range(len(w) - 3):
                    features.append(w[i:i+4])
        return features

    def _build_vector_space(self):
        """Constructs IDF weights and normalized anchor vectors."""
        all_docs = []
        for anchor_list in self.action_anchors.values():
            all_docs.append(" ".join(anchor_list))

        num_docs = len(all_docs)
        doc_features = [set(self._extract_ngrams(d)) for d in all_docs]

        all_unique_features = set().union(*doc_features)
        self.vocab = {feat: idx for idx, feat in enumerate(sorted(all_unique_features))}

        # Calculate IDF
        for feat in self.vocab:
            df = sum(1 for df_set in doc_features if feat in df_set)
            self.idf[feat] = math.log((num_docs + 1) / (df + 1)) + 1.0

        # Pre-compute TF-IDF vector for each Action Anchor Cluster
        self.anchor_vectors: Dict[str, Dict[str, float]] = {}
        for action, doc in zip(self.action_anchors.keys(), all_docs):
            self.anchor_vectors[action] = self._vectorize(doc)

    def _vectorize(self, text: str) -> Dict[str, float]:
        """Computes normalized TF-IDF vector for text."""
        ngrams = self._extract_ngrams(text)
        if not ngrams:
            return {}

        tf: Dict[str, float] = {}
        for ng in ngrams:
            if ng in self.vocab:
                tf[ng] = tf.get(ng, 0.0) + 1.0

        norm_sq = 0.0
        vec: Dict[str, float] = {}
        for ng, count in tf.items():
            tfidf = (1.0 + math.log(count)) * self.idf.get(ng, 1.0)
            vec[ng] = tfidf
            norm_sq += tfidf * tfidf

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 1.0
        return {ng: val / norm for ng, val in vec.items()}

    def _cosine_similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Computes dot product of two normalized vectors."""
        if not vec_a or not vec_b:
            return 0.0
        # Iterate over smaller dict
        small, large = (vec_a, vec_b) if len(vec_a) < len(vec_b) else (vec_b, vec_a)
        score = sum(val * large.get(ng, 0.0) for ng, val in small.items())
        return max(0.0, min(1.0, score))

    def route_intent(self, user_prompt: str) -> Dict[str, Any]:
        """
        Classifies user prompt into the most semantically proximate agent action.
        Returns top action, confidence score, and ranked candidate scores.
        """
        query_vec = self._vectorize(user_prompt)
        scores = []

        for action, anchor_vec in self.anchor_vectors.items():
            sim = self._cosine_similarity(query_vec, anchor_vec)
            scores.append((action, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_action, top_score = scores[0]

        return {
            "primary_action": top_action,
            "confidence": round(top_score, 4),
            "candidates": [(a, round(s, 4)) for a, s in scores[:3]]
        }

if __name__ == "__main__":
    router = SemanticIntentRouter()
    test_cases = [
        "make a portfolio website for a web developer, save all the file in ATUL folder",
        "create a modern portfolio website inside this portfolio folder",
        "analys this project",
        "analyse project architecture and imports",
        "make a folder name ATUL in this current working directory",
        "plan how to implement user authentication with JWT in FastAPI",
        "list all files in this directory",
        "fastapi database connection error how to fix"
    ]

    print("=" * 80)
    print("  DALI-1 ZERO-REGEX SEMANTIC INTENT ROUTER VERIFICATION")
    print("=" * 80)
    for q in test_cases:
        res = router.route_intent(q)
        print(f"Prompt:     '{q}'")
        print(f"Top Action:  {res['primary_action']} (Confidence: {res['confidence']})")
        print(f"Candidates:  {res['candidates']}")
        print("-" * 80)
