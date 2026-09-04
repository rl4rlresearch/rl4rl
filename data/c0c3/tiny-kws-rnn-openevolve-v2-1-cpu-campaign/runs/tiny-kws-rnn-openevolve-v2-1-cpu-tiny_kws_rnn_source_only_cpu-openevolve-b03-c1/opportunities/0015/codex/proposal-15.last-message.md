MECHANISM: Conservative recurrent width reduction with trajectory readout

HYPOTHESIS: An 88-unit full-rate GRU using both final state and temporal mean will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.8% versus the successful 92-unit model.

INTENDED_EDIT: Reduce the GRU hidden width, temporal summary, and classifier input from 92 to 88 units while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 92-unit endpoint-and-trajectory model achieved 86.50% accuracy, slightly exceeding the 96-unit model’s 86.26%; this suggests the enriched readout retains enough capacity for another modest width reduction without discarding temporal evidence.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(184, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE