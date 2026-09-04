MECHANISM: Full-resolution GRU width pruning with dual-timescale readout

HYPOTHESIS: A 104-unit GRU using all 32 frames and concatenated mean/final states will retain at least 85% accuracy while reducing total inference MACs by approximately 11.3% versus the qualified 111-unit design.

INTENDED_EDIT: Reduce recurrent width from 128 to 104, preserve the full causal frame schedule, and classify from both the temporal mean and final recurrent output.

EVIDENCE: The 111-unit dual-readout design achieved 87.85% accuracy with substantial margin, while both 24-frame designs failed; this supports preserving temporal resolution and testing whether the richer readout permits further width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
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