Completed the configured run.

- Run directory: [sol-xhigh-20260816](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260816)
- Attempts recorded: **200/200**
- Baseline confirmed recorded first: **6,080 parameters, 100.00%**
- Final retained frontier: **1,417 parameters, 99.03%**
- Reduction: **4,663 parameters (76.69%)**
- Final commit: `553f9516e9e7deed44b63d56a58f41296dc07460`
- The official runner subsequently reported: `Attempt budget exhausted; no command was run.`

Experiment accounting:

- 112 regular attempts and 88 automation macros
- 761 automation micro-trials
- Macro outcomes: 95 keep, 87 discard, 18 error
- Candidate outcomes: 726 retained, 125 scored nonqualifications, 22 errors
- Macro allocation: parameter tying 98; scalar pruning 30; feed-forward width 22; attention organization 21; position representation 10; token representation 9; normalization 7; training control 3
- Fresh-training logs record 925,000 aggregate steps across 27 runs and 259.1 reported training minutes. The full ledger spans 36h 24m.

The trajectory combined interleaved LSB-first representation, width/projection reductions, shared QKV and feed-forward transforms, and extensive scalar grouping across embeddings, attention, FF, and normalization tensors. Important local boundaries included the positional union that failed at 1,792 parameters/98.97%, and an FF-output union at 1,434/98.91%; later cross-tensor changes reopened other eligible pairs. Attempt 200 ended at its cap with three successes, so it did not establish a lower bound.

Audit note: attempt 183’s narrative summary says four accepts ending at 1,482, but its five official micro-rows and macro fields correctly record five accepts ending at 1,481. The final state and retained commit are consistent. Tracked candidate files are clean; only allowed automation helpers remain untracked.

Ledgers: [RESULTS.tsv](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260816/RESULTS.tsv), [AUTOMATION_RESULTS.tsv](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260816/AUTOMATION_RESULTS.tsv), [STATE.json](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260816/STATE.json).

This is only the best frontier observed in this bounded run—not a global optimum or lower bound. Per [PROGRAM.md](/Users/utshaho/Documents/GitHub/rl4rl/data/raw/autoresearch/sol-xhigh-20260816/workspace/PROGRAM.md:199), independent final-holdout evaluation remains required; the reported accuracy is the fixed-seed development qualification signal.