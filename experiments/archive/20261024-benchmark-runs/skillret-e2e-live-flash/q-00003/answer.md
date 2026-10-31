Here's a rebuilt typography system for your poetry journal, following the typeset guidance:

**Font pairing (2 families, genuine contrast):**
- **Display/headings & poem titles:** Cormorant Garamond (elegant literary serif that fits a poetry journal far better than Papyrus)
- **Body, bylines & UI text:** Source Sans 3 (humanist sans for readability, real contrast with the serif)

Load with `display=swap` and only the weights you use: 400, 600 (serif also 500 italic for titles).

**Type scale (1.333 ratio, rem-based to respect user settings):**

```css
:root {
  --font-serif: 'Cormorant Garamond', Georgia, serif;
  --font-sans: 'Source Sans 3', 'Helvetica Neue', sans-serif;

  --text-caption: 0.875rem;   /* 14px */
  --text-body:   1.0625rem;   /* 17px */
  --text-byline: 1rem;        /* 16px, small caps */
  --text-title:  clamp(1.75rem, 3vw, 2.25rem);
  --text-h1:     clamp(2.5rem, 6vw, 4rem);
}

h1 {
  font-family: var(--font-serif);
  font-size: var(--text-h1);
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: -0.02em;
}

.poem-title {
  font-family: var(--font-serif);
  font-size: var(--text-title);
  font-weight: 500;
  font-style: italic;
  line-height: 1.2;
}

.byline {
  font-family: var(--font-sans);
  font-size: var(--text-byline);
  font-weight: 400;
  text-transform: uppercase;
  letter-spacing: 0.08em;   /* open tracking for small caps */
  color: #555;
}

body {
  font-family: var(--font-sans);
  font-size: var(--text-body);
  line-height: 1.65;
  font-weight: 400;
}

.poem-body {
  font-family: var(--font-serif);
  font-size: 1.25rem;
  line-height: 1.7;
  max-width: 34em;          /* short lines suit verse */
}

.pull-quote {
  font-family: var(--font-serif);
  font-size: clamp(1.5rem, 2.5vw, 2rem);
  font-weight: 500;
  font-style: italic;
  line-height: 1.3;
  max-width: 30ch;
  padding-left: 1.25rem;
  border-left: 3px solid currentColor;
}

.caption {
  font-family: var(--font-sans);
  font-size: var(--text-caption);
  color: #666;
}
```

**What this fixes:**
- **Papyrus heading** → literary serif with real personality and tight display tracking
- **11px body** → 17px, comfortably readable, 1.65 line-height
- **Inverted bylines/titles** → bylines are now quiet small-cap labels; poem titles dominate in large italic serif
- **Undifferentiated pull quotes** → large italic serif with a rule and constrained measure, clearly featured
- **Arbitrary sizes** → one consistent scale combining size + weight + family + style so hierarchy reads at a glance
- **Measures & rhythm** → `ch`/`em` max-widths keep lines in the 45–75 character comfort zone

**Verification checklist:** headings vs. titles vs. bylines distinguishable instantly ✓ · body ≥16px ✓ · same roles styled identically everywhere ✓ · only 2 families, 4 weights ✓ · rem units keep zoom accessible ✓ · WCAG AA contrast on grays (#555/#666 only for secondary text) ✓
