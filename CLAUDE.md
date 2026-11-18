# CLAUDE.md

This file tells Claude how to work in this folder.

## What this folder is

This is not a code project. It holds interview-preparation notes about one job on a resume:
a contract role at Dell Technologies (October 2024 to March 2025) on a sales prediction AI
platform built on AWS.

The main file is `dell.md`. It explains, in plain language:

- what the project was and why Dell needed it,
- how the data flowed (Kafka streaming and batch loads, then ETL, then an S3 data lake, a
  vector store and a Neo4j graph database, then RAG and AI agents),
- what the person actually did (the ETL data pipelines only, not the AI part),
- how each resume bullet relates to the project,
- warnings where the resume and the real work do not match,
- likely interview questions with suggested answers.

## Rules for editing the notes

Keep the content honest. The person said they only worked on ETL. Do not write anything that
claims they built or led the AI, RAG or agent work. Where the resume says more than they did,
point it out and suggest an honest way to say it.

Keep the same plain, friendly style that `dell.md` already uses. Explain every technical term
the first time it appears, as if the reader is new to the field.

## How we work in this project

This adds to the "How we work" rules in my global CLAUDE.md and wins where they differ.

- The plan is in `specs/` (`spec.md`, `plan.md`, `tasks.md`). Follow `tasks.md` in order.
- Each task is split into subtasks (T0.1, T0.2, ...). **One subtask is one commit.** Finish one
  subtask, report what changed, what ran and the result, give the commit message, then wait.
- Commit messages use Conventional Commits: `docs: ...`, `feat: ...`, `fix: ...`, `test: ...`,
  `refactor: ...`, `chore: ...`.
- I manage all Git work myself. Do not run Git write commands, and do not comment on Git setup,
  branches or how to group commits. Just give the commit message.

## How to write to me

Always give me context that is fully understandable and friendly to a human reader.

- Use short, simple sentences with one idea each. Do not write long, dense sentences that pack
  in many facts.
- Use everyday words. When a technical word is needed, explain it in a few plain words.
- Start with the main point, then explain it.
- Write in normal paragraphs, the way you would explain something out loud.
- If you are not sure the explanation is clear, make it simpler rather than shorter.
