MECHANISM: Dual-timescale recurrent readout with slight width pruning

HYPOTHESIS: A 111-unit full-resolution GRU with a learned classifier over both mean and final recurrent states will retain at least 85% accuracy while reducing total MACs by about 1.6% versus the qualified 112-unit design.

INTENDED_EDIT: Reduce the GRU width from 128 to 111, retain all 32 causal frames, and concatenate the temporal mean with the final hidden state for classification.

EVIDENCE: Full-frame 112-unit recurrence achieved 85.15%, whereas 24-frame variants failed even at 128 units; this motivates preserving every frame and recovering discriminative capacity through a richer low-cost readout rather than further temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        _hidden, summary, count = state
        return self.classifier(summary / count.clamp_min(1.0))
=======
        hidden, summary, count = state
        mean_output = summary / count.clamp_min(1.0)
        final_output = hidden[:, 0, :]
        return self.classifier(torch.cat((mean_output, final_output), dim=-1))
>>>>>>> REPLACE