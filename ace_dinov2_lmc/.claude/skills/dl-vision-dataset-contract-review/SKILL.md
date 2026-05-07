---
name: dl-vision-dataset-contract-review
description: Use when auditing dataset layout, annotations, split conventions, scene naming, transforms, or dataloader assumptions in a deep learning or vision repository. Treat the dataset interface as a strict contract and surface mismatches early.
---

# DL / Vision Dataset Contract Review

Use this skill when a project depends on nontrivial dataset structure, custom loaders, scene folders, metadata files, or annotation formats.

## Workflow

1. Identify all dataset entrypoints.
   Find dataset classes, backend selectors, scene-root logic, and split routing.
2. Record the contract.
   Required files, directory names, metadata files, annotation formats, image modalities, and naming assumptions.
3. Record transform behavior.
   Resolution, crop, augmentation, normalization, dtype conversion, and mask semantics.
4. Record sampling behavior.
   Per-scene, per-image, per-view, sequential, random, weighted, or replacement-based sampling.
5. Compare code assumptions against real directories.
   Detect missing files, renamed scenes, inconsistent IDs, or undocumented split rules.
6. Produce a validation checklist.
   Short commands or file checks that can confirm the dataset is compatible before training.

## Output

- dataset contract summary
- required path/layout checklist
- transform/sampling summary
- mismatches or ambiguities
- preflight validation steps

## Rules

- Treat dataset root resolution as part of the public API.
- Distinguish between backend-specific logic and generic loader logic.
- If multiple backends exist, document them separately.
- Prefer exact path examples and exact expected filenames.
