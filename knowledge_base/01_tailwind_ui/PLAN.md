# Module 1: Tailwind CSS & Modern UI Engineering Plan

> **Objective:** Train DALI-1 to master Tailwind CSS utility classes, responsive mobile-first layouts, and modern component synthesis (Hero, Navbar, Cards, Forms, Modals).

---

## 1. Official Documentation Topics to Ingest
1. **Core Layout Utilities:**
   - Flexbox: `flex`, `flex-col`, `items-center`, `justify-between`, `gap-4`
   - Grid: `grid`, `grid-cols-1`, `md:grid-cols-3`, `gap-6`, `col-span-2`
   - Container & Box Alignment: `container`, `mx-auto`, `max-w-7xl`, `px-4`
2. **Spacing & Sizing:**
   - Width & Height: `w-full`, `max-w-md`, `h-screen`, `min-h-[400px]`
   - Padding & Margin: `p-6`, `py-12`, `px-4`, `m-auto`, `-mt-8`
3. **Typography & Styling:**
   - Font sizes & weights: `text-xs`, `text-sm`, `text-xl`, `text-4xl`, `font-bold`
   - Colors & Opacity: `text-slate-900`, `text-gray-500`, `text-indigo-600`
4. **Borders, Shadows & Effects:**
   - Rounded corners: `rounded-lg`, `rounded-full`, `rounded-2xl`
   - Shadows: `shadow-sm`, `shadow-md`, `shadow-xl`, `ring-1 ring-black/5`
5. **Interactive Pseudo-Classes & Dark Mode:**
   - Hover & Focus: `hover:bg-indigo-700`, `focus:ring-2 focus:ring-offset-2`
   - Transitions: `transition-all duration-200 ease-in-out`
   - Dark Mode: `dark:bg-slate-900 dark:text-white`
6. **Production Component Synthesis:**
   - Modern Landing Page Hero Section with Call to Action
   - Responsive Navigation Bar with mobile hamburger toggle
   - Feature 3-Column Card Grid with hover lift effects
   - Pricing Table with Popular tier highlighted
   - Contact Form with floating labels & accessible inputs

---

## 2. Verification Gates
- **Gate 1:** Dataset contains clean `<|user|>` / `<|assistant|>` pairs with valid Tailwind utility classes.
- **Gate 2:** Tokenizer successfully parses all Tailwind specific prefixes (`md:`, `lg:`, `hover:`, `ring-1`).
- **Gate 3:** Warm-start training converges with perplexity $< 1.10$.
- **Gate 4:** CLI prompt `"Generate a responsive Hero section with Tailwind CSS"` outputs clean, valid HTML/JSX code.
