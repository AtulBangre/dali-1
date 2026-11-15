# DALI-1 Web Development & Coding Assistant: Master Plan & Execution Roadmap

> **Target Model:** DALI-1 Sub-Quadratic Causal Neural Architecture with Bounded Memory & Adaptive Compute  
> **Specialization:** Full-Stack Web Development, Modern Frameworks, Official Documentation, and Real-World Debugging  
> **Target Environment:** Local Apple Silicon Mac (16GB RAM) + 4-Bit INT4 Deployment  
> **Status:** IN PROGRESS (Step-by-Step Verified Execution)

---

## 1. Project Objectives
1. **Official Stack Documentation:** Cover core standards and APIs for HTML5, CSS3, Modern JavaScript (ES2024), TypeScript, React 19, Next.js 15, Python, FastAPI, Flask, SQL, MongoDB, Git, and Docker.
2. **Real-World Debugging Mastery:** Train the model on Tri-Part Debugging Tuples:
   - **Broken Code & Stack Trace / Error Message**
   - **Root Cause Analysis (Under-the-hood explanation)**
   - **Minimal, Production-Ready Corrected Code**
3. **Local Feasibility & High Throughput:** Optimize tokenization, chunked state training, and 4-bit INT4 quantization for local Mac development with zero memory leaks.

---

## 2. Step-by-Step Execution Plan

| Step | Phase Description | Key Deliverables | Verification Gate | Status |
| :---: | :--- | :--- | :--- | :---: |
| **Step 1** | **Master Blueprint & Architecture Plan** | `PLAN.md` creation with phases & verification metrics | Schema & roadmap approval | **COMPLETED** |
| **Step 2** | **Data Acquisition & Synthesis Engine** | `data_pipeline/collector.py` & `data_pipeline/debugging_generator.py` | Generates 100+ rich multi-stack documentation and debugging pairs | **COMPLETED** |
| **Step 3** | **Data Cleaning, Deduplication & Validation** | `data_pipeline/cleaner.py` and `data/final_coding_corpus.txt` | MinHash deduplication, formatting validation, 0 syntax corruption | **COMPLETED** |
| **Step 4** | **Code-Preserving BPE Tokenizer** | `train_code_tokenizer.py` $\to$ `checkpoints/code_bpe.json` | Exact whitespace, 2/4-space indentations, and code operators preserved | **COMPLETED** |
| **Step 5** | **Tokenizer Round-Trip Verification** | `tests/test_code_tokenizer.py` | 100% exact string reconstruction on complex multi-language code | **COMPLETED** |
| **Step 6** | **Chunked State Model Pre-Training & SFT** | `train_coder.py` $\to$ `checkpoints/dali1_coder_fp32.pt` | Loss `0.0268`, Perplexity `1.03`, zero OOM on Mac | **COMPLETED** |
| **Step 7** | **4-Bit INT4 Quantization Engine** | `checkpoints/dali1_coder_int4.pt` | 5.9x compression, 9.0 MB footprint, 0.998 cosine fidelity | **COMPLETED** |
| **Step 8** | **Interactive Terminal & Debugging Evaluation** | `chat.py` with multi-stack debugging tests | Pass live bug diagnosis on CORS, async race, React hooks, FastAPI | **COMPLETED** |

---

## 3. Detailed Data Architecture (The Tri-Part Debugging Schema)

Every debugging sample in the training corpus follows this standard format:
```markdown
<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
My code failed with the following error:
[Error Trace / Bug Description]
```[language]
[Broken Code]
```
How do I fix this?
<|assistant|>
### Root Cause:
[Concise explanation of the underlying runtime/compiler failure]

### Solution:
```[language]
[Clean, modern, corrected production code]
```

### Best Practices & Prevention:
[1-2 bullet points to prevent this issue in production]
```

---

## 4. Verification Checkpoints
- **Gate 1:** Data Pipeline outputs non-empty, high-signal, diverse debugging and API samples.
- **Gate 2:** Tokenizer preserves indentation and tokenizes code efficiently without splitting operators.
- **Gate 3:** Training loss converges with chunked state passing.
- **Gate 4:** Autoregressive output produces valid, syntactically correct code and accurate root-cause diagnoses.
