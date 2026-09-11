MECHANISM: Width-trimmed 16-step causal GRU

HYPOTHESIS: Retaining the qualified 16-frame even-index schedule while reducing GRU width from 128 to 124 will preserve at least 85% validation accuracy and reduce recurrent MACs by approximately 5.7% versus Reference Design 1.

INTENDED_EDIT: Process every second frame and consistently reduce recurrent, summary, and classifier width to 124 units.

EVIDENCE: The 16-step 128-unit design achieved 85.153% accuracy, whereas reducing temporal coverage to 15 steps fell to 80.245%; this motivates preserving all 16 proven time samples and testing a modest orthogonal width reduction.

<<<<<<< SEARCH
GRAD_CLIP_NORM = 1.0
=======
GRAD_CLIP_NORM = 1.0
HIDDEN_SIZE = 124
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, HIDDEN_SIZE, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(HIDDEN_SIZE, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(
            batch_size, 1, HIDDEN_SIZE, device=device, dtype=dtype
        )
        summary = torch.zeros(
            batch_size, HIDDEN_SIZE, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return list(range(available_frames))
=======
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE