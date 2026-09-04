MECHANISM: Disentangled mean–concentration channel attention

HYPOTHESIS: Replacing the correlated maximum descriptor with max-minus-mean concentration will exceed 9,162 correct predictions by separating distributed activation strength from localized salience while preserving parameter count, identity initialization, and two gate evaluations.

INTENDED_EDIT: Feed the shared channel gate average-pooled features and the parameter-free max-minus-mean concentration descriptor, then retain equal signal fusion.

EVIDENCE: Mean-only attention reached 9,153 correct and mean–maximum attention reached 9,162, showing spatial extrema add useful information; the three-descriptor concentration variant timed out, so substituting concentration for the redundant raw maximum tests that signal without its extra gate evaluation.

<<<<<<< SEARCH
        average_context = features.mean(dim=(2, 3))
        maximum_context = features.amax(dim=(2, 3))
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
        )
=======
        average_context = features.mean(dim=(2, 3))
        maximum_context = features.amax(dim=(2, 3))
        concentration_context = maximum_context - average_context
        channel_signal = 0.5 * (
            self.channel_gate(average_context)
            + self.channel_gate(concentration_context)
        )
>>>>>>> REPLACE