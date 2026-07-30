# Job monitoring & ops

Monitor jobs directly on the cluster with the standard SLURM tools (`squeue -u $USER`, `sacct`,
the `logs/%x-%j.{out,err}` files each job script writes). The bespoke monitoring/ledger tooling
used during development is internal and not part of this release.
Command verification before unattended launches is covered in
[COMMAND_AUDIT.md](COMMAND_AUDIT.md).
