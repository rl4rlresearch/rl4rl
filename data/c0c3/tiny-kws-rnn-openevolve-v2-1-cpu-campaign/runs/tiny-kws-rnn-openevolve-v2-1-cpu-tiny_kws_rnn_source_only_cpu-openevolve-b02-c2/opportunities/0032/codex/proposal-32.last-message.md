MECHANISM: Adjacent recurrent-width refinement at the qualified 29-step schedule

HYPOTHESIS: An 84-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 633,935,525 to approximately 620,521,440.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier dimensions from 91 to 84 while preserving the qualified schedule and training procedure.

EVIDENCE: The 85-unit design achieved 86.99% validation accuracy with 633,935,525 MACs, leaving a 1.99-point margin and making width 84 the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(182, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 84, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(168, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 84, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 84, device=device, dtype=dtype)
>>>>>>> REPLACE