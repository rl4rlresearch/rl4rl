# Repository instructions for coding agents

## Complete human goals autonomously

- A human-authored goal authorizes every ordinary, in-scope action needed to
  complete and verify that goal. If it says to start, run, launch, resume, or
  continue a campaign or process, that is the required start authorization.
  Do not pause later for a redundant confirmation.
- Assume the human may be away after assigning a goal. Continue across
  prerequisite work, tests, validation gates, launch, monitoring, recovery,
  and final verification instead of stopping at an intermediate milestone.
- The authorization persists across automatic goal continuations, context
  compaction, and later agent turns. Reaching a consequential step that the
  goal explicitly requested does not create a new confirmation requirement.
- Resolve ordinary ambiguity from repository evidence, current state, prior
  operator choices, and reasonable in-scope defaults. Ask for input only when
  the goal explicitly requests a hold or choice, essential information or
  authority cannot be inferred, or an external permission boundary makes
  progress impossible.
- Treat safety, integrity, validation, and recovery requirements as work to
  perform autonomously, not as reasons to seek duplicate approval. Invoke any
  required tool-level approval directly and cite the original human
  instruction rather than first asking for another chat confirmation.

More specific `AGENTS.md` files may add local engineering and scientific
requirements. Explicit human instructions take precedence over conflicting
repository-authored guidance.
