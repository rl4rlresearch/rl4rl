MECHANISM: Fixed mean-biased dual-statistic channel attention

HYPOTHESIS: A 60/40 mean-to-maximum gate mixture will exceed 9,162 correct predictions by retaining complementary maximum context while emphasizing the stronger average-pooled descriptor.

INTENDED_EDIT: Replace equal channel-signal fusion with a parameter-free mixture favoring average-pooled context, preserving initialization, parameter count, and runtime.

EVIDENCE: Equal mean–maximum fusion achieved 9,162 correct, while the symmetric 40/60 max-biased mixture fell to 9,155 and mean-only attention achieved 9,153; testing the unexplored mean-biased direction is the most informative refinement of the best mechanism.

<<<<<<< SEARCH
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
=======
        channel_signal = (
            0.6 * self.channel_gate(average_context)
            + 0.4 * self.channel_gate(maximum_context)
        )
>>>>>>> REPLACE