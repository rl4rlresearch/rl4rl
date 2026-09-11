MECHANISM: Conservative recurrent-width boundary refinement

HYPOTHESIS: A 110-unit GRU processing the qualified 30-frame schedule will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.3% versus the qualified 112-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 128 to 110 while preserving the 30-frame causal schedule and training procedure.

EVIDENCE: The 112-unit, 30-step model achieved 85.276% accuracy, and the reduction from 120 to 112 units cost only 0.491 percentage points; a smaller two-unit reduction directly probes the remaining width boundary with less risk than another eight-unit step.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(110, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
        count = torch.zeros(batch_size, 1, device=device, dtype=dtype)
>>>>>>> REPLACE