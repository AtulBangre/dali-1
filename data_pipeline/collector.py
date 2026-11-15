import os
from typing import List
from data_pipeline.debugging_generator import generate_debugging_corpus

OFFICIAL_DOCS_DATA = [
    # React 19 Standards
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
Explain the React 19 Server Components paradigm and when to use 'use client'.
<|assistant|>
### React 19 Server Components vs Client Components:
1. **Server Components (Default):**
   - Execute strictly on the server and send zero JavaScript to the browser bundle.
   - Ideal for direct database queries, backend API calls, and static content rendering.
   - Cannot use browser hooks (`useState`, `useEffect`, `useContext`) or DOM event handlers (`onClick`).
2. **Client Components ('use client'):**
   - Declared by placing `'use client';` at the top of the file.
   - Necessary whenever you need interactive event listeners (`onClick`, `onChange`), state management, browser APIs (`window`, `localStorage`), or custom React hooks.""",

    # TypeScript Type Narrowing
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
How do discriminated unions work in TypeScript for robust type checking?
<|assistant|>
### Discriminated Unions in TypeScript:
A discriminated union uses a shared literal property (the 'discriminant') to enable TypeScript to narrow down the exact type within conditional branches.

```typescript
type LoadingState = { status: 'loading' };
type SuccessState = { status: 'success'; data: string[] };
type ErrorState = { status: 'error'; message: string };

type NetworkState = LoadingState | SuccessState | ErrorState;

function handleState(state: NetworkState) {
    switch (state.status) {
        case 'loading':
            return 'Loading...';
        case 'success':
            return `Data: ${state.data.join(', ')}`;
        case 'error':
            return `Failed: ${state.message}`;
    }
}
```
This guarantees compile-time exhaustiveness and prevents runtime property access errors.""",

    # Next.js 15 Route Handlers
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
How do you write a Next.js 15 App Router Route Handler with JSON validation?
<|assistant|>
### Next.js 15 Route Handler (`app/api/users/route.ts`):
```typescript
import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url);
    const role = searchParams.get('role') || 'all';

    return NextResponse.json({
        success: true,
        data: [{ id: 1, name: 'Alice', role }]
    }, { status: 200 });
}

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        if (!body.email || !body.name) {
            return NextResponse.json(
                { success: false, error: 'Name and email are required' },
                { status: 400 }
            );
        }

        return NextResponse.json(
            { success: true, user: { id: Date.now(), ...body } },
            { status: 201 }
        );
    } catch {
        return NextResponse.json({ success: false, error: 'Invalid JSON' }, { status: 400 });
    }
}
```""",

    # FastAPI Dependency Injection & Auth
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
How do you implement API authentication in FastAPI using Depends and HTTPBearer?
<|assistant|>
### FastAPI Dependency Injection Authentication:
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

app = FastAPI()
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    if token != "secret-production-token":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return "authorized-user"

@app.get("/api/protected-resource")
def get_protected_data(user: str = Depends(verify_token)):
    return {"status": "success", "message": f"Hello {user}, access granted!"}
```""",

    # SQL Indexing & Performance
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
When should you create a composite index in PostgreSQL and how does column order matter?
<|assistant|>
### PostgreSQL Composite Indexing:
A composite (multi-column) index indexes queries filtering on multiple columns simultaneously:
```sql
CREATE INDEX idx_orders_user_created ON orders (user_id, created_at DESC);
```
### Column Order Rule (Leftmost Prefix):
PostgreSQL can only use the index if your query conditions match from left to right:
- `WHERE user_id = 5 AND created_at >= '2025-01-01'` -> **Uses full index**.
- `WHERE user_id = 5` -> **Uses index (leftmost prefix matched)**.
- `WHERE created_at >= '2025-01-01'` -> **Cannot use index efficiently** because the leading column (`user_id`) is omitted.""",

    # CSS Modern Grid Layout
    """<|system|>
You are DALI-1, an expert Software Engineer and Full-Stack Web Development Assistant.
<|user|>
Write a production-ready responsive card grid in CSS with auto-fit and minmax.
<|assistant|>
### Modern Responsive Grid (No Media Queries Required):
```css
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 300px), 1fr));
    gap: 1.5rem;
    padding: 1rem;
}

.card {
    display: flex;
    flex-direction: column;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    overflow: hidden;
}
```
`minmax(min(100%, 300px), 1fr)` automatically wraps columns on smaller screens without any horizontal overflow."""
]

def build_coding_corpus(output_path: str = "data/coding_corpus.txt") -> int:
    os.makedirs("data", exist_ok=True)
    debugging_samples = generate_debugging_corpus()
    all_samples = debugging_samples + OFFICIAL_DOCS_DATA

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n<|end_block|>\n\n".join(all_samples))

    print(f"✓ Successfully compiled {len(all_samples)} high-signal coding samples into {output_path}")
    return len(all_samples)

if __name__ == "__main__":
    build_coding_corpus()
