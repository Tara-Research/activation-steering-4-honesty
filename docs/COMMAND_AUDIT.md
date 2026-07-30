# COMMAND_AUDIT.md — verify every runbook command before automation runs it

## Why this file exists

Any automation that builds launch/diagnose commands straight from [RUNBOOK.md](RUNBOOK.md) and
[REPRODUCTION.md](REPRODUCTION.md) will hit a runtime failure on a stale command. Before a stage's
commands are trusted for unattended launch, they must be verified *runnable*.

Command/doc consistency is maintained by hand: verify flags against the argparse definitions /
shell-script case statements before editing the docs.

## How to audit one command
1. Run it at the smallest scope (`--test_mode` / `--test`, a single `--techniques`, one
   `--scenario_filter`).
2. Confirm jobs submit (`squeue`) and the expected results path appears.
3. Only then let automation prepare/launch that stage within the usual approval gates.
