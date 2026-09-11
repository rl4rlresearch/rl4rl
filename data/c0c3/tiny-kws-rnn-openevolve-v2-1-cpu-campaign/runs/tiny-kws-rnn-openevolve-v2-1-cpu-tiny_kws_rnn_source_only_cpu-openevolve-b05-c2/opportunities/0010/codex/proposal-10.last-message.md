MECHANISM: Boundary-seeking GRU width pruning with dual-timescale readout

HYPOTHESIS: An 82-unit full-resolution GRU with concatenated mean/final states will retain at least 85% validation accuracy while reducing total inference MACs by approximately 4.3% versus the qualified 84-unit design.

INTENDED_EDIT: Reduce the GRU and summary width from 96 to 82 and resize the classifier input from 192 to 164, while preserving all 32 causal frames and the established training procedure.

EVIDENCE: The 84-unit design achieved 87.73% accuracy, while 80 units achieved 84.29%; testing 82 units directly probes the remaining width boundary with substantially more accuracy margin than linear interpolation requires.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 82, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(164, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 82, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 82, device=device, dtype=dtype)
>>>>>>> REPLACE