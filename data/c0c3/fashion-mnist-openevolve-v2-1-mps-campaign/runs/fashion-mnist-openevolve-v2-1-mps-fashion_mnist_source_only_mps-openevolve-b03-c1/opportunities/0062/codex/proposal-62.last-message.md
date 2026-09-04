MECHANISM: Shared mean–maximum–concentration channel attention

HYPOTHESIS: Explicitly gating the max-minus-mean channel concentration descriptor will exceed 9,162 correct predictions by distinguishing broad texture responses from isolated salient activations while preserving identity initialization and the parameter count.

INTENDED_EDIT: Add a parameter-free spatial concentration descriptor to the successful shared channel gate and equally combine its signal with the existing average- and maximum-pooled signals.

EVIDENCE: Mean-only attention reached 9,153 correct and adding maximum context raised this to 9,162, showing complementary spatial statistics help; max-minus-mean exposes their otherwise implicit interaction without the runtime-heavy backbone pooling that timed out.

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
        channel_signal = (
            self.channel_gate(average_context)
            + self.channel_gate(maximum_context)
            + self.channel_gate(concentration_context)
        ) / 3.0
>>>>>>> REPLACE