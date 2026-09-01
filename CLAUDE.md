# CLAUDE.md

This file is the Claude entrypoint for repository guidance. The canonical agent
instructions, repository orientation, and workflow expectations live in
[AGENTS.md](AGENTS.md); full contribution and PR requirements live in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Project working preferences

The following summary is kept here so Claude-facing workflows apply the same
defaults. `AGENTS.md` remains authoritative when details change.

### Engineering

- Preserve behavior during refactors. Trace complete call paths, compare every
  affected backend and task with the baseline, and document and test every
  intentional difference.
- Prefer small, explicit, composable abstractions with clear ownership and
  lifecycle rules. Make invalid states difficult to represent, and avoid dynamic
  machinery when a direct constructor or method is easier to understand.
- Design public APIs for developers who do not know the scheduler or hardware
  internals. Names, accepted input types, return types, and constructor forms
  should be clear from type hints and concise docstrings.
- Judge an abstraction by how it extends and composes with existing components
  in both local and remote configurations. Review the entire affected surface,
  not only the most recent implementation slice.
- Verify changes with focused regressions, reusable contract tests, mock SDKs,
  parity checks, and end-to-end tests as appropriate.
- Comments and docstrings explain contracts, invariants, ownership, and
  non-obvious reasons. They do not narrate the code or make non-informative
  claims about dependencies the code does not use.

### Documentation and communication

- Use natural, professional technical language in docs, comments, PRs, reviews,
  and conversation. Avoid casual chat, bureaucratic phrasing, promotional prose,
  and formulaic AI-like cadence.
- Treat continuity as a basic requirement for every article and substantive
  explanation, not only code documentation. Establish the reader's situation,
  result, scope, and reading order at the start; connect each section to the
  preceding state; make paragraphs depend on one another; and frame and
  interpret examples. Explain interface operations in caller order, including
  the relevant inputs, returns, and lifecycle effects.
- Explain the reader-visible situation before naming the implementation. Teach
  the normal workflow first, followed by extension, composition, remote use, and
  internal architecture only when each layer becomes relevant.
- Examples should complete a real workflow: compose the new component with an
  existing one, show how it is read or controlled, and connect it to a task or
  environment.
- Check every technical statement against code. Explain related types, return
  values, ownership, and lifecycle rules before relying on them in examples or
  headings.
- Write English directly. Write Chinese according to natural Chinese logic,
  using restrained written language and full-width punctuation. Keep familiar
  developer terms in English when translation would be unusual or harder to
  search; in RL prose, keep `policy` in English.
- Keep English and Chinese pages equivalent in meaning and structure without
  translating sentence by sentence. Follow [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md)
  for the complete terminology, structure, and RST rules.
