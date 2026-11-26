---
level: Contributor guide
runtime: MkDocs Material
description: Maintain NeuroGebra's website, generated demonstrations, page structure and accessibility.
---

# Maintain the documentation

The website uses one MkDocs Material build. A dedicated homepage introduces the library; task-oriented guides and focused reference pages support deeper reading. Existing documentation URLs remain available.

## Design and page structure

Warm paper backgrounds, ink text, teal accents and serif headings connect the site to a mathematical notebook. Shared styles live in `docs/stylesheets/neurogebra.css`. Keep decorative effects secondary to equations, code and results. Check both light and dark themes.

The homepage template is `overrides/home.html`. Shared article metadata and the announcement are in `overrides/main.html`. The activation and training components are separate template partials. Avoid copying a separate header/search implementation: the site retains Material's navigation and search.

For a tutorial, include a short outcome, prerequisites, a complete example, expected results, the mathematics, input shapes, common mistakes and the next guide. Optional front matter adds a compact context row:

```yaml
level: Foundations
runtime: NumPy / CPU
description: A short explanation of what this page helps the reader do.
```

## Reproduce the illustrations

```bash
python scripts/generate_site_data.py
```

This writes `docs/assets/data/explorer.json` and `docs/assets/images/sigmoid-preview.svg`. Function and derivative samples come from MathForge. The recorded loss trace comes from a seeded 80-epoch Model fit with the Observatory enabled. No benchmark timings or external datasets are generated.

The browser only inspects precomputed samples. It does not execute Python, compile kernels or train a model. The ReLU note explains its nondifferentiable point and the symbolic convention used at zero. Keep these distinctions visible when extending the demo.

The SVG, Python example and recorded training values remain available without JavaScript. A failed data request leaves that fallback in place. Browser interaction code lives in `docs/javascripts/home-demo.js`; it loads demonstration data only when the homepage component is present.

## Navigation and equations

The site's five documentation areas are Get started, Guides, Examples, Reference and Project. Keep labels short and organize guides by the task a reader wants to complete. Core API classes have focused pages; `api/core.md` retains legacy anchors and links to the new references.

Material instant navigation updates page content without a full reload. MathJax is loaded only on pages containing equations, and rendering runs again after navigation. Initialize new components through Material's document lifecycle and prevent duplicate event listeners. Preserve the correct code annotation feature flag, `content.code.annotate`.

## Validate changes

```bash
python -m mkdocs build --strict
python scripts/check_docs.py --site site
python -m pytest tests/test_site_data.py -q
node --test tests/web/home-demo.test.cjs
python -m mkdocs serve
```

Use a browser to check narrow mobile, tablet and desktop widths, both color schemes, keyboard navigation, search, clipboard behavior, demo controls, training tabs, mathematical rendering and back/forward navigation. Native code blocks and wide tables may scroll inside their containers; ordinary page content should fit the viewport. Respect reduced-motion preferences and keep visible focus indicators.

When adding a new template or asset directory, include it in the source distribution and deployment triggers. Keep generated outputs current and include them in the same change as their generator.

## Implementation validation

The November 26, 2026 update was checked with the full Python suite (519 passed, 8 optional/native checks skipped), five JavaScript interaction tests, a strict documentation build, runnable guide introductions and the built-site link checker. All 88 legacy core API anchors are retained.

Sixteen browser checks covered desktop, 768px and 360px layouts, light/dark styling, search, clipboard feedback, keyboard tabs, activation controls, navigation, legacy reference links, no-JavaScript behavior and failed data requests. Equation-loader lifecycle tests used a controlled renderer; live MathJax CDN loading could not be exercised in this execution environment.
