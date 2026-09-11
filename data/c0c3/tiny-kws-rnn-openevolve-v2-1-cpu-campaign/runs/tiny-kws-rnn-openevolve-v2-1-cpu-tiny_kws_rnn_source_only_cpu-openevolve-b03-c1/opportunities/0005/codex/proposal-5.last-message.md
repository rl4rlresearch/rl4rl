MECHANISM: Conservative full-rate recurrent width reduction

HYPOTHESIS: A 108-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by about 6.5% versus the successful 112-unit model.

INTENDED_EDIT: Reduce the GRU hidden state and classifier input from 112 to 108 units while preserving the full causal frame schedule and training procedure.

EVIDENCE: The 112-unit full-rate GRU achieved 85.77% accuracy, whereas reducing temporal coverage to 28 frames failed at 83.93%; this supports preserving all frames and making a smaller structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 108, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(108, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 108, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 108, device=device, dtype=dtype)
>>>>>>> REPLACE