# Open-Source Figure Workflow

This file records how open-source figure / prompt resources should shape our method-figure preparation before project-specific details are added.

## Useful Open-Source References

### PaperBanana

Repository: https://github.com/llmsresearch/paperbanana

Use as a high-level academic-figure workflow reference:

- Start from a text description of the method.
- Convert it into an optimized diagram instruction.
- Generate a draft.
- Evaluate and refine the draft.
- Treat the output as a publication-quality target or a reference for vector recreation.

Project adaptation:

- We use the same staged workflow, but keep the source-of-truth method definition in `figure_spec.md`.
- We do not rely on generated figures as final paper assets unless labels and arrows are manually verified.

### GPT-Image2-Skill

Repository: https://github.com/wuyoscar/GPT-Image2-Skill

Use for prompt organization:

- State the intended use first, e.g. academic method figure / infographic diagram.
- Use a consistent order: background or canvas, subject, key details, constraints.
- For research figures, treat generated images as references, workflow sketches, or reproducible style targets.
- Keep dense text controlled and explicitly list the required labels.

Project adaptation:

- `image2_prompt.md` uses a structured prompt with sections for canvas, layout, labels, style, and constraints.
- Required method labels are listed explicitly to reduce text drift.

### visual-skills

Repository: https://github.com/smixs/visual-skills

Use for visual-prompt discipline:

- Classify this task as an infographic / diagram / slide.
- Prefer GPT Image 2 style prompting for small dense text, multi-font text, and structured diagrams.
- Specify typography, spacing, and text-rendering constraints.

Project adaptation:

- The prompt requests clean sans-serif text, readable labels, thin arrows, and no decorative content.
- Dense implementation details such as optimizer, GPU, batch size, and run directories are explicitly excluded.

### Mermaid Skills

Reference: https://github.com/Agents365-ai/mermaid-skill

Use for structural drafting:

- Build a text-based graph first.
- Validate nodes and arrows before visual styling.
- Keep the diagram editable and reproducible.

Project adaptation:

- `method_architecture.mmd` is the first structural draft.
- Image generation should follow the Mermaid structure rather than inventing a new pipeline.

## Project-Specific Layer

After applying the open-source workflow, the diagram must reflect our method:

- Offline scene memory construction from posed training images and geometry.
- Compact scene memory / map-code tokens.
- GeoLMC compressor and memory-to-image fusion.
- ACE / GLACE encoder producing dense local features and a GLACE global descriptor.
- Stage 1 local memory-conditioned ACE coordinate head.
- Stage 2 GLACE concatenation / global head.
- DSAC* / RANSAC pose solving from refined scene coordinates.

## Policy

Generated images are design references. The final paper figure should be checked against `figure_spec.md` and preferably recreated or cleaned in a vector editor if exact text, arrows, or module placement matter.
