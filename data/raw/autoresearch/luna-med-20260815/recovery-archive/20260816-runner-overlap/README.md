# Luna runner-overlap recovery

The previous runner lacked an exclusive lock. Multiple active invocations used
the same attempt IDs, leaving duplicate `RESULTS.tsv` rows and attempt folders
that no longer matched their recorded result rows.

All regular-attempt artifacts, prior state/results, raw agent logs, and
orphaned temporary directories from that collision have been moved here
unchanged for audit. They are not valid experiment evidence and are excluded
from the repaired run.

The repaired run retains only the independently verified 6,080-parameter
baseline and restarts regular attempt numbering at `attempt-0001`.
