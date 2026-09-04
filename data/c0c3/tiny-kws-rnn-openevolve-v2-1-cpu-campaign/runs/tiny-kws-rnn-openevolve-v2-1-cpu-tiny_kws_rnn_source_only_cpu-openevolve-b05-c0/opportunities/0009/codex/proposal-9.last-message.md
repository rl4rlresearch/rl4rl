MECHANISM: Recurrent-width scaling with dual temporal readout

HYPOTHESIS: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 14.6% versus the verified 96-unit model.

INTENDED_EDIT: Reduce the GRU hidden width from 96 to 88 and resize the classifier and recurrent summary state accordingly.

EVIDENCE: The 96-unit full-sequence model achieved 87.98% accuracy, retaining a 2.98-point margin after the successful reduction from 104 units; width scaling has preserved accuracy more reliably than temporal subsampling.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE