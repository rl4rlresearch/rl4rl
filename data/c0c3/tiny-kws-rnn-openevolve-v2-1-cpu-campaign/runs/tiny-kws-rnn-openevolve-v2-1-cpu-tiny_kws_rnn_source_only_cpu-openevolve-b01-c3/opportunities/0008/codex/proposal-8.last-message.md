MECHANISM: Further width trimming of the qualified 16-step causal GRU

HYPOTHESIS: A 112-unit GRU trained with batch size 64 on the proven even-index schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.3% versus the qualified 116-unit design.

INTENDED_EDIT: Reduce recurrent, summary-state, and classifier width from the current 124 units to 112 while preserving the successful training regime and 16-frame schedule.

EVIDENCE: The 116-unit design achieved 86.50% accuracy, 1.50 points above the requirement, after successive four-unit reductions from 124 to 120 to 116 all qualified with batch size 64.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
>>>>>>> REPLACE