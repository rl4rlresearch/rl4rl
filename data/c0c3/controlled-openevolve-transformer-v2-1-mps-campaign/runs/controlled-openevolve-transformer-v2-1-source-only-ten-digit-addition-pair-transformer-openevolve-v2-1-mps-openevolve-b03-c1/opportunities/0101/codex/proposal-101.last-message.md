MECHANISM: Shared gain with disjoint head-slot routing

HYPOTHESIS: Tying the two learned head gains will reduce the verified model from 1,046 to 1,045 parameters while retaining at least 99% accuracy, because the successful design shows that preserving separate residual slots is load-bearing, whereas both routes already share their value representation and should not require independent scalar amplitudes.

INTENDED_EDIT: Replace the two per-head output gains with one learned RMS energy-matched gain, while keeping each attention head’s context in its existing distinct residual-channel slot.

EVIDENCE: Summing the head contexts before projection collapsed accuracy to 0.01%, but disjoint fixed routing with per-head gains achieved 99.96% at 1,046 parameters. This isolates route identity as essential and makes scalar amplitude tying the smallest direct test of whether head-specific output scaling is also necessary.

<<<<<<< SEARCH
            gain = torch.linalg.vector_norm(
                head_maps, dim=(0, 2)
            ) / math.sqrt(self.head_dim)
            self.proj_gain = nn.Parameter(gain.clone())
=======
            gain = torch.linalg.vector_norm(head_maps) / math.sqrt(d_model)
            self.proj_gain = nn.Parameter(gain.reshape(1).clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y * self.proj_gain.view(1, self.n_head, 1, 1)
=======
        y = y * self.proj_gain.view(1, 1, 1, 1)
>>>>>>> REPLACE