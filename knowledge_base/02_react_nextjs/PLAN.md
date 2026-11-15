# Module 2: React 19 & Next.js 15 Full-Stack Engineering Plan

> **Objective:** Master modern React 19 and Next.js 15 App Router architecture, Server Actions, Hydration boundaries, and Component composition.

## 1. Key Topics & Standards
1. **React 19 Foundations:**
   - Server Components vs Client Components (`'use client'`).
   - Server Actions (`'use server'`) for form mutations without API routes.
   - Core Hooks: `useState`, `useEffect`, `useMemo`, `useCallback`, `useTransition`, `useOptimistic`.
2. **Next.js 15 App Router:**
   - Special Files: `page.tsx`, `layout.tsx`, `loading.tsx`, `error.tsx`, `not-found.tsx`.
   - Data Fetching: Parallel and sequential data fetching with caching (`cache: 'force-cache'`, `revalidate`).
   - Route Handlers: `app/api/.../route.ts` with NextRequest and NextResponse.
   - Dynamic Routes: `[slug]/page.tsx` and catch-all `[...segments]/page.tsx`.
3. **Common Pitfalls & Debugging:**
   - Preventing Hydration mismatches.
   - Avoiding state updater calls in render phase.
   - Passing serializable props across the Server-Client boundary.
