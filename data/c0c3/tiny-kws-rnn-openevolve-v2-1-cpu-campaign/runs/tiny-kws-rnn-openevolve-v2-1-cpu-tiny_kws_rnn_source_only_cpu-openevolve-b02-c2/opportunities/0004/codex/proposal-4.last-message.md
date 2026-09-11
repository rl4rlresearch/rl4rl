MECHANISM: Full-resolution recurrent width reduction

HYPOTHESIS: A 112-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by 12% versus the qualified 120-unit design.

INTENDED_EDIT: Reduce the GRU state, temporal summary, and classifier width from 128 to 112 while preserving every input frame and the established training procedure.

EVIDENCE: The 120-unit full-resolution GRU achieved 86.50% accuracy, whereas reducing temporal steps caused both tested models to fail; this motivates retaining all 32 frames and testing the remaining capacity margin through a moderate width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE