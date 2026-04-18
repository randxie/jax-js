# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Dev Environment

- Always use the virtual environment located at `./.venv`.
- Run scripts using `.venv/bin/python` to ensure dependencies are loaded.
- Any file downloaded from the internet that is not meant to be sent back to the user must be placed in `.download`. Create the folder if it does not exist.
- Use `.artifacts/local/` for regular generated artifacts that are only for local inspection, debugging, or intermediate workflow state. Create folder if not exist.
- Use `.artifacts/send/` only for artifacts that are explicitly intended to be sent back to the user or Telegram as deliverables. Create folder if not exist.
- Not every artifact belongs in the send-back path. Most generated artifacts should stay local unless the user explicitly asked to receive them.
- Never use `.artifacts/send/` to send back the whole repository, source trees, project folders, or bulk code exports. `.artifacts/send/` is only for small explicit deliverables.
- Keep send-back artifacts as top-level files directly under `.artifacts/send/`. Do not place directories or nested trees there.
- Send back at most 1 artifact per turn. If multiple local artifacts are generated, choose the single intended deliverable and keep the rest in `.artifacts/local/`.
- Do not send downloaded files back to the user just because they exist locally.
- Do not trigger the backward attachment path unless the user explicitly asked for a deliverable artifact to be sent back. Requests to make code changes, fix bugs, update UI, write docs for review, or inspect files are not permission to send attachments back.
- Only send artifacts back to the user when the user explicitly requested generated output or explicitly asked to receive a file. If a file should be attached back to the user, make sure it is present in `.artifacts/send/` first, then send it as an attachment.

## 6. Code Merging

- When user asks you to upload the code, make sure you properly update .gitignore so you don't accidentally push artifacts that cause storage issues. After checking that, create a commit message based on the change, and push the change to main. Return proper error if you failed to push. Never do force push!!
