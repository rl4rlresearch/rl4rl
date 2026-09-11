MECHANISM: Adjacent recurrent-width refinement

HYPOTHESIS: A 91-unit dual-view GRU processing frames 1–29 will retain at least 85% validation accuracy while reducing total dense inference MACs from 731,804,800 to approximately 717,398,045.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and dual-view classifier dimensions from 92 to 91 while preserving the qualified 29-frame schedule and training procedure.

EVIDENCE: The 92-unit, 29-step design achieved 85.64% validation accuracy with 731,804,800 MACs; its positive margin makes the adjacent 91-unit width the closest unresolved structural reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 91, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(182, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 91, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 91, device=device, dtype=dtype)
>>>>>>> REPLACE