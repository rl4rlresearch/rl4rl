MECHANISM: Recurrent-width-to-spectral-context reallocation

HYPOTHESIS: An 83-unit GRU using mel bands 1–19 on frames 4–28 will retain at least 85% validation accuracy while reducing total inference MACs from 519,679,860 to approximately 518,566,570.

INTENDED_EDIT: Replace one recurrent hidden unit with two additional spectral inputs, discard only the lowest mel band, and adopt the qualified 25-step schedule while retaining the dual-view classifier.

EVIDENCE: The 84-unit, 17-band, 25-step model qualified at 85.03%, while an 84-unit, 19-band model reached 86.75% on the longer schedule and bands 1–17 outperformed bands 0–16. This motivates preserving upper-band evidence while exploiting the quadratic hidden-state cost to reduce MACs.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 85, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(170, 8)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 83, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(166, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 85, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 85, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 83, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 83, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(frame[..., 1:20]).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(frames[..., 1:20]),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(1, available_frames - 2))
=======
        return list(range(4, available_frames - 3))
>>>>>>> REPLACE