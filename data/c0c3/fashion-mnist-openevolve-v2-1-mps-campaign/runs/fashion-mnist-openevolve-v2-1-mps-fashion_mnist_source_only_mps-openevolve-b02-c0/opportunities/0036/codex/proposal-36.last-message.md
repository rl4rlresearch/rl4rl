MECHANISM: Dense terminal-checkpoint weight averaging

HYPOTHESIS: Averaging equally many snapshots over only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by reducing BatchNorm mismatch without sacrificing the variance reduction of the successful final-5% average.

INTENDED_EDIT: Start parameter averaging at 97.5% progress and double its sampling frequency from every four steps to every two steps, preserving approximately the current number of averaged snapshots.

EVIDENCE: Narrowing averaging from the final 10% to 5% retained 9,252 correct and reduced cross-entropy from 0.2360583 to 0.2356299; the prior 2.5% attempt retained the four-step cadence and timed out, so preserving snapshot count is the most targeted next test.

<<<<<<< SEARCH
    if progress >= 0.95 and (step % 4 == 0 or is_final_step):
=======
    if progress >= 0.975 and (step % 2 == 0 or is_final_step):
>>>>>>> REPLACE