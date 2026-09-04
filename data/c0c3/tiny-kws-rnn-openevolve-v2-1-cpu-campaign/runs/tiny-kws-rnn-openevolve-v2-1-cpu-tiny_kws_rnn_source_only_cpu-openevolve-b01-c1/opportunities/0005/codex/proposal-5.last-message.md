MECHANISM: Full-resolution GRU width compression

HYPOTHESIS: A 96-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 13.7% versus the verified 104-unit design.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 104 to 96 while preserving the full causal frame schedule and training procedure.

EVIDENCE: The 104-unit full-resolution GRU achieved 86.01% accuracy, slightly improving on the 112-unit model’s 85.89%, while both temporal-subsampling experiments failed; this supports another width reduction without discarding frames.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(96, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE