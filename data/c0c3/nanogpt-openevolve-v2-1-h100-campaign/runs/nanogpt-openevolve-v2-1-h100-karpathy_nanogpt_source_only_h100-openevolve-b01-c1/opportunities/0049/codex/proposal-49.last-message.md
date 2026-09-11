MECHANISM: Lower-sensitivity bounded value-residual gating

HYPOTHESIS: Halving the gate’s local sensitivity will reduce val_bpb below 0.983689 by slowing deviations from its neutral value while retaining input-dependent routing and the same bounded output range.

INTENDED_EDIT: Reparameterize the per-head value-residual gate from `2*sigmoid(z)` to `1+tanh(z/4)`, preserving zero-init output 1 and bounds between 0 and 2.

EVIDENCE: Doubling sensitivity with `1+tanh(z)` regressed val_bpb from 0.983689 to 0.984585; testing the opposite direction is the most informative adjacent change.

<<<<<<< SEARCH
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
=======
            gate = 1 + torch.tanh(0.25 * self.ve_gate(x[..., :self.ve_gate_channels]))
>>>>>>> REPLACE