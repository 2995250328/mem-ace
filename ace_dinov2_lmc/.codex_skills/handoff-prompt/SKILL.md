---
name: handoff-prompt
description: Use when the user asks to create, update, refresh, or write a handoff prompt for this project. Always overwrite the repository handoff prompt file and return its absolute path.
metadata:
  short-description: Write project handoff prompts
---

# Handoff Prompt

When the user asks for a handoff prompt, transition prompt, server handoff, or asks to update the handoff area:

1. Write the handoff prompt to:
   `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/HANDOFF_PROMPT.md`
2. Overwrite the file rather than appending.
3. Include the concrete current state, files to read first, completed experiments, current code status, validation already run, next command(s), acceptance checks, and cautions.
4. Keep the prompt directly copyable into a new Codex conversation.
5. After writing, always return the absolute path to the file in the final answer.

Before writing, inspect the current progress docs if needed:

- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/EXECUTION_PLAN_reference_consistent_memory.md`
- `/home/xwh/project/ace_depth/ace_dinov2_lmc/memory_extraction/03_implementation/PROGRESS_reference_consistent_memory.md`

Do not create a new handoff file unless the user explicitly asks for a different path.
