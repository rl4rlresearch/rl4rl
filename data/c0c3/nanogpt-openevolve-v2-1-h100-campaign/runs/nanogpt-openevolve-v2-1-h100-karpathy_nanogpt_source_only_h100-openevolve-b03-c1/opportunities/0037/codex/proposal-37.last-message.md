MECHANISM: Full-state value-residual routing

HYPOTHESIS: Expanding each value-embedding gate from 32 input channels to the full 512-dimensional hidden state will reduce val_bpb below 0.984227 without materially affecting throughput.

INTENDED_EDIT: Let the existing per-head value gate condition on every normalized hidden-state channel instead of an arbitrary 32-channel prefix.

EVIDENCE: Replacing the learned value gate with a fixed neutral mixture regressed val_bpb from 0.984227 to 0.986944 while preserving essentially the same throughput (495.5M versus 495.0M tokens), showing that learned value-residual routing is useful and its small projection is not a meaningful compute bottleneck.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
=======
        self.ve_gate_channels = self.n_embd
>>>>>>> REPLACE