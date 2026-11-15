import os
import json
from typing import List, Dict

DEBUGGING_DATA: List[Dict[str, str]] = [
    # 1. React / Next.js Hydration & Hooks
    {
        "category": "React / Next.js",
        "error": "Error: Hydration failed because the initial UI does not match what was rendered on the server.",
        "broken_code": """// app/page.tsx
export default function Page() {
    return (
        <div>
            <h1>Current Time: {new Date().toLocaleTimeString()}</h1>
        </div>
    );
}""",
        "root_cause": "The server and client render different values because `new Date().toLocaleTimeString()` executes at two different timestamps, causing server-client HTML mismatch.",
        "solution": """// app/page.tsx
'use client';
import { useEffect, useState } from 'react';

export default function Page() {
    const [time, setTime] = useState<string | null>(null);

    useEffect(() => {
        setTime(new Date().toLocaleTimeString());
    }, []);

    return (
        <div>
            <h1>Current Time: {time || 'Loading...'}</h1>
        </div>
    );
}""",
        "prevention": "Never access dynamic, client-specific APIs (Date, window, localStorage) directly during server-side initial rendering. Defer them to `useEffect`."
    },
    {
        "category": "React Hooks",
        "error": "Error: Too many re-renders. React limits the number of renders to prevent an infinite loop.",
        "broken_code": """function UserCounter() {
    const [count, setCount] = useState(0);
    // Bug: Calling state updater directly in render phase
    setCount(count + 1);
    return <div>{count}</div>;
}""",
        "root_cause": "Calling `setCount()` directly in the component body triggers a re-render before the current render completes, causing an infinite loop.",
        "solution": """function UserCounter() {
    const [count, setCount] = useState(0);

    const increment = () => {
        setCount(prev => prev + 1);
    };

    return <button onClick={increment}>Count: {count}</button>;
}""",
        "prevention": "State setters must only be invoked inside event handlers or `useEffect` hooks with appropriate dependency arrays."
    },
    {
        "category": "JavaScript / Web APIs",
        "error": "Access to fetch at 'https://api.example.com/data' from origin 'http://localhost:3000' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present.",
        "broken_code": """// frontend/app.js
async function getData() {
    const res = await fetch("https://api.example.com/data");
    const json = await res.json();
    return json;
}""",
        "root_cause": "The backend server does not include the `Access-Control-Allow-Origin` header in its HTTP response to authorize cross-origin browser requests.",
        "solution": """# backend/main.py (FastAPI)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/data")
def get_data():
    return {"status": "success", "data": [1, 2, 3]}""",
        "prevention": "Configure CORS middleware explicitly on the backend server for all allowed frontend domains."
    },
    {
        "category": "JavaScript / Async",
        "error": "TypeError: Cannot read properties of undefined (reading 'map')",
        "broken_code": """function UserList({ users }) {
    return (
        <ul>
            {users.map(u => <li key={u.id}>{u.name}</li>)}
        </ul>
    );
}""",
        "root_cause": "The `users` prop is undefined during the initial render before asynchronous data loading completes.",
        "solution": """function UserList({ users = [] }) {
    if (!users || users.length === 0) {
        return <p>No users found.</p>;
    }

    return (
        <ul>
            {users.map(u => (
                <li key={u.id}>{u.name}</li>
            ))}
        </ul>
    );
}""",
        "prevention": "Always provide default fallback values (`users = []`) or use optional chaining (`users?.map(...)`)."
    },
    {
        "category": "Python / FastAPI",
        "error": "pydantic_core._pydantic_core.ValidationError: 1 validation error for Item price: Field required [type=missing]",
        "broken_code": """from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
def create_item(item: Item):
    return item""",
        "root_cause": "The client payload sent in the request body omitted the required `price` field.",
        "solution": """from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class Item(BaseModel):
    name: str
    price: Optional[float] = 0.0  # Default value or Optional

@app.post("/items")
def create_item(item: Item):
    return {"status": "created", "item": item}""",
        "prevention": "Define optional fields with `Optional[T] = None` or provide sensible default values in Pydantic models."
    },
    {
        "category": "Python / Async",
        "error": "RuntimeError: Task <Task pending> got Future <Future pending> attached to a different loop",
        "broken_code": """import asyncio

async def fetch_item():
    return {"id": 1}

def run_sync():
    # Bug: Creating a new loop while one is already running
    return asyncio.get_event_loop().run_until_complete(fetch_item())""",
        "root_cause": "`run_until_complete()` cannot be called when an event loop is already active in asynchronous frameworks like FastAPI or Jupyter.",
        "solution": """import asyncio

async def fetch_item():
    return {"id": 1}

async def run_proper():
    result = await fetch_item()
    return result""",
        "prevention": "In async environments, await coroutines directly instead of trying to synchronously block the running event loop."
    },
    {
        "category": "CSS / Layout",
        "error": "Layout Bug: Horizontal scrollbar appears on mobile screen and page shakes horizontally.",
        "broken_code": """.hero-section {
    width: 100vw;
    margin: 0;
    padding: 20px;
    box-sizing: content-box;
}""",
        "root_cause": "`100vw` includes the width of the vertical scrollbar, and `box-sizing: content-box` adds padding onto 100vw, exceeding viewport width.",
        "solution": """* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

.hero-section {
    width: 100%;
    max-width: 100%;
    padding: 20px;
    overflow-x: hidden;
}""",
        "prevention": "Use `width: 100%` instead of `100vw`, set `box-sizing: border-box` globally, and ensure parents clip unintended overflow."
    },
    {
        "category": "SQL / Databases",
        "error": "OperationalError: FATAL: remaining connection slots are reserved for non-replication superuser connections",
        "broken_code": """# Bug: Creating a new database connection on every HTTP request without closing
def get_user(user_id):
    conn = psycopg2.connect(DB_URI)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    return cursor.fetchone()""",
        "root_cause": "Creating unpooled raw connections per request exhausts PostgreSQL `max_connections` limit.",
        "solution": """from psycopg2 import pool

connection_pool = pool.SimpleConnectionPool(minconn=1, maxconn=20, dsn=DB_URI)

def get_user(user_id):
    conn = connection_pool.getconn()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, name FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
    finally:
        connection_pool.putconn(conn)""",
        "prevention": "Always use connection pooling (SQLAlchemy engine, Prisma, or psycopg2 pool) with context managers to release connections."
    },
    {
        "category": "Docker / DevOps",
        "error": "Docker build takes 10+ minutes on every minor code change because `npm install` runs every time.",
        "broken_code": """FROM node:20
WORKDIR /app
COPY . .
RUN npm install
CMD ["npm", "start"]""",
        "root_cause": "`COPY . .` copies all source files before `RUN npm install`, invalidating Docker's layer cache on any file edit.",
        "solution": """FROM node:20-alpine
WORKDIR /app
# 1. Copy package files first to leverage Docker layer caching
COPY package*.json ./
RUN npm ci --only=production

# 2. Copy application source code after dependencies are installed
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]""",
        "prevention": "Separate dependency manifests (`package.json` / `requirements.txt`) from application code to preserve Docker build caching."
    },
    {
        "category": "Git / Version Control",
        "error": "error: failed to push some refs to 'origin/main'. Updates were rejected because the remote contains work that you do not have locally.",
        "broken_code": """# Attempting to force push blindly
git push origin main""",
        "root_cause": "Remote branch has commits that your local branch lacks. A simple push is rejected to protect history.",
        "solution": """# 1. Fetch remote changes and rebase local commits cleanly
git fetch origin
git rebase origin/main

# 2. If there are merge conflicts, resolve them and continue:
# git add <resolved-files>
# git rebase --continue

# 3. Push cleanly
git push origin main""",
        "prevention": "Always pull or rebase before pushing (`git pull --rebase origin main`) and avoid using `--force` on shared branches."
    }
]

def generate_debugging_corpus() -> List[str]:
    formatted_samples = []
    for d in DEBUGGING_DATA:
        sample = f"""<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
Category: {d['category']}
My code failed with the following error:
{d['error']}

Broken Code:
{d['broken_code']}

How do I fix this?
<|assistant|>
### Root Cause:
{d['root_cause']}

### Solution:
{d['solution']}

### Prevention:
{d['prevention']}"""
        formatted_samples.append(sample.strip())
    return formatted_samples

if __name__ == "__main__":
    samples = generate_debugging_corpus()
    print(f"Generated {len(samples)} comprehensive debugging cases.")
