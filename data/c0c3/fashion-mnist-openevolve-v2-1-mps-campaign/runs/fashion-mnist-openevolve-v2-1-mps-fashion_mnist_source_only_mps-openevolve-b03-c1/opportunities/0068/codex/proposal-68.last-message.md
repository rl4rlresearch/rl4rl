MECHANISM: Local max-biased dual-statistic channel attention

HYPOTHESIS: A 45/55 mean-to-maximum gate mixture will exceed 9,162 correct predictions by preserving the successful near-equal fusion while modestly favoring the empirically less harmful maximum-biased direction.

INTENDED_EDIT: Replace equal channel-signal fusion with a parameter-free 45/55 mixture favoring maximum-pooled context.

EVIDENCE: Equal fusion achieved 9,162 correct; the 40/60 max-biased variant retained 9,155 versus 9,152 for the symmetric 60/40 mean-biased variant, motivating a midpoint probe between the best mixture and the stronger neighboring direction.

<<<<<<< SEARCH
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
=======
        channel_signal = (
            0.45 * self.channel_gate(average_context)
            + 0.55 * self.channel_gate(maximum_context)
        )
>>>>>>> REPLACE