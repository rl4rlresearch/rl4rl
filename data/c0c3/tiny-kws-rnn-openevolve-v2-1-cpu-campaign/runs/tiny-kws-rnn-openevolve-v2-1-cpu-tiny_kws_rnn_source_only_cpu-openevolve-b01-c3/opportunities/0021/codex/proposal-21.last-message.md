MECHANISM: One-unit recurrent width boundary trim

HYPOTHESIS: A 100-unit GRU will retain at least 85% validation accuracy while reducing total inference MACs from 478,744,040 to approximately 470,092,000.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 104 to 100 units while preserving the qualified batch-64 training procedure and 16-frame even-index schedule.

EVIDENCE: The adjacent 101-unit design qualified at 85.276% accuracy and 478,744,040 MACs; 100 units is the smallest untested structural reduction and isolates the next cost boundary without changing temporal coverage or optimization.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
>>>>>>> REPLACE