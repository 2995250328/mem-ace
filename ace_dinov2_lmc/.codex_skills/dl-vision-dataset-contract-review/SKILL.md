---
name: dl-vision-dataset-contract-review
description: Use when auditing dataset layout, annotations, split conventions, scene naming, transforms, or dataloader assumptions in a deep learning or vision repository. Treat the dataset interface as a strict contract and surface mismatches early.
---

# DL / Vision Dataset Contract Review

Use this skill when a project depends on nontrivial dataset structure, custom loaders, scene folders, metadata files, or annotation formats.

## Workflow

1. Identify all dataset entrypoints.
2. Record the contract.
3. Record transform behavior.
4. Record sampling behavior.
5. Compare code assumptions against real directories.
6. Produce a validation checklist.
