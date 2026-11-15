# DALI-1 Knowledge Base & Multi-Stack Curriculum Master Index

This directory organizes official documentation, curriculum plans, and specialized datasets for training DALI-1 into an autonomous, full-stack software engineering assistant.

## Tech Stack Modules

| Directory | Stack Name | Status | Key Deliverables |
| :--- | :--- | :---: | :--- |
| `00_html5_standards/`| **HTML5 Semantic Web & DOM** | **COMPLETED** | W3C semantic layouts, accessible forms, native details/dialog |
| `01_tailwind_ui/` | **Tailwind CSS & Full Utility** | **COMPLETED** | Complete utility encyclopedia, responsive design, components |
| `02_react_nextjs/` | **React 19 & Next.js 15** | **COMPLETED** | Server Components, App Router, Hooks, Server Actions |
| `03_mern_backend/` | **Node.js, Express & MongoDB** | **COMPLETED** | REST APIs, JWT Auth Middleware, Mongoose Schemas & CRUD |
| `04_fastapi_postgres/`| **FastAPI & PostgreSQL** | **COMPLETED** | Async Python Endpoints, Pydantic v2, SQLAlchemy ORM |
| `05_docker_devops/` | **Docker & Git Workflows** | **COMPLETED** | Multi-stage Dockerfiles, Compose, Git Rebase & Branching |

## Execution Protocol
For each stack:
1. `PLAN.md`: Topic breakdown, API references, and target components.
2. `docs/`: Official documentation markdown extracts.
3. `curated_dataset.txt`: Instruction-tuned training pairs (`<|user|>` / `<|assistant|>`).
4. Incremental warm-start training without catastrophic forgetting.
5. Verification via CLI generation test.
