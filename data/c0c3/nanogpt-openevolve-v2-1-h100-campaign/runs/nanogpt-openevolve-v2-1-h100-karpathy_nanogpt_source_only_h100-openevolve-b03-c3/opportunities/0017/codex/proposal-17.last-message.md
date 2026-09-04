MECHANISM: Bounded-amplitude attention-head gating

HYPOTHESIS: Preserving the proven gate’s neutral slope while limiting each head’s multiplier to 0.5–1.5 will prevent excessive head suppression or amplification and lower val_bpb below 0.993637 without reducing throughput.

INTENDED_EDIT: Replace the 0–2 sigmoid output gate with a neutral, equal-slope 0.5–1.5 tanh gate; batching, conditioning channels, softcap, and optimization remain unchanged.

EVIDENCE: The original first-32-channel sigmoid gate achieved the best val_bpb of 0.993637, while increasing its neutral slope to 0.75 regressed to 0.996193. Matching the successful 0.5 initial slope while narrowing only the attainable amplitude isolates whether large gate excursions caused that regression.

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
        y = y * out_gate.unsqueeze(-1)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 1 + 0.5 * torch.tanh(self.out_gate(x[..., :self.ve_gate_channels]))
        y = y * out_gate.unsqueeze(-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Gate weights init to zero (sigmoid(0)=0.5, scaled by 2 -> 1.0 = neutral)
=======
        # Gate weights init to zero (both gate parameterizations evaluate to 1.0)
>>>>>>> REPLACE