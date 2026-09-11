MECHANISM: Fixed max-biased dual-statistic channel attention

HYPOTHESIS: A 40/60 mean-to-maximum gate mixture will exceed 9,162 correct predictions by modestly extending the observed gain from adding localized maximum context, without the instability of learned fusion weights.

INTENDED_EDIT: Replace equal channel-signal fusion with a parameter-free mixture favoring maximum-pooled context while preserving identity initialization and parameter count.

EVIDENCE: Mean-only attention achieved 9,153 correct and equal mean–maximum fusion improved this to 9,162, whereas learned per-channel fusion fell to 9,141 and learned global fusion timed out; a fixed max-biased mixture is the cleanest next probe.

<<<<<<< SEARCH
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
=======
        channel_signal = (
            0.4 * self.channel_gate(average_context)
            + 0.6 * self.channel_gate(maximum_context)
        )
>>>>>>> REPLACE