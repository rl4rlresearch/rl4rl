MECHANISM: Incremental recurrent-width compression

HYPOTHESIS: Reducing the verified 28-step GRU from 92 to 91 hidden units will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 2%.

INTENDED_EDIT: Keep the successful frames 4–31 schedule and training procedure, but structurally reduce the GRU state and classifier input width to 91.

EVIDENCE: The 92-unit, 28-step model achieved 85.40%, while both tested 27-step schedules failed; this motivates preserving temporal coverage and probing the smallest possible dense recurrent-width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
=======
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(91, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
>>>>>>> REPLACE