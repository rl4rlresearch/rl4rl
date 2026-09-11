MECHANISM: Lower-overhead short-horizon sparse EMA

HYPOTHESIS: A 0.985-decay EMA sampled every 16 steps will exceed 9,334 correct predictions by tracking late convergence more closely while avoiding the timeout encountered by the eight-step 0.985 experiment.

INTENDED_EDIT: Shorten the successful EMA horizon from roughly 100 to 67 optimizer steps and halve its snapshot frequency from every eight steps to every sixteen.

EVIDENCE: The current sparse 0.99 EMA improved the same unaveraged model from 9,322 to 9,334 correct; the proposed 0.985 horizon was unverified only because training timed out, motivating a lower-overhead implementation of that localized EMA test.

<<<<<<< SEARCH
    should_average = progress >= 0.5 and (
        step % 8 == 0 or step >= total_steps
    )
=======
    should_average = progress >= 0.5 and (
        step % 16 == 0 or step >= total_steps
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
                    decay = 0.99 ** elapsed
=======
                    decay = 0.985 ** elapsed
>>>>>>> REPLACE