# Module 00: HTML5 Semantic Web & DOM Standards Plan

> **Objective:** Train DALI-1 on the complete W3C / MDN HTML5 specification, covering every structural element, form control, accessibility pattern, and metadata tag.

## Core Knowledge Areas
1. **Document Metadata & Shell:**
   - `<!DOCTYPE html>`, `<html lang="en">`, `<head>`, `<meta charset="UTF-8">`, `<meta name="viewport" content="width=device-width, initial-scale=1.0">`, `<title>`, `<link rel="icon">`.
2. **Semantic Structural Layout:**
   - `<header>`: Page/section header containing branding, titles, search bars.
   - `<nav>`: Primary navigation links with `<ul role="list">`.
   - `<main>`: Unique, non-repeated primary content of the document.
   - `<section>`: Thematic grouping of content with an explicit heading (`<h2>`-`<h6>`).
   - `<article>`: Self-contained, independently distributable composition (blog post, card, news item).
   - `<aside>`: Tangentially related content, sidebars, related links.
   - `<footer>`: Author information, copyright, legal links, sitemaps.
3. **Modern Form Controls & Accessibility (ARIA):**
   - `<form action="..." method="...">`: Form submission wrapper.
   - `<label for="input-id">`: Associated label binding for screen readers.
   - Inputs: `type="text"`, `type="email"`, `type="password"`, `type="number"`, `type="checkbox"`, `type="radio"`, `type="file"`, `type="date"`.
   - `<textarea rows="..." cols="...">`, `<select>` with `<option value="...">`, `<button type="submit">`.
   - ARIA: `aria-label`, `aria-expanded`, `aria-hidden="true"`, `role="alert"`, `role="dialog"`.
4. **Rich Media & Graphics:**
   - `<img>` with `src`, `alt` (mandatory for a11y), `loading="lazy"`, `srcset`.
   - `<picture>` with responsive `<source media="..." srcset="...">`.
   - `<video controls poster="...">` with `<track kind="subtitles">`.
   - `<audio controls>`, `<svg>`, `<canvas>`.
5. **Interactive & Modern UI Elements:**
   - `<details>` and `<summary>` for native accordion dropdowns without JavaScript.
   - `<dialog>` with `.showModal()` and `.close()` for native modal popups.
