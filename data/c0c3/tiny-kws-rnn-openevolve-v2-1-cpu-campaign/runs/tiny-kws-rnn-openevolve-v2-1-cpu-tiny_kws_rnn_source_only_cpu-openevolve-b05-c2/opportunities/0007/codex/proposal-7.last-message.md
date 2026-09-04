MECHANISM: Full-resolution GRU width pruning with dual-timescale readout

HYPOTHESIS: An 88-unit GRU processing all 32 frames and classifying concatenated mean/final states will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 14.7% versus the qualified 96-unit design.

INTENDED_EDIT: Reduce the GRU hidden and summary width from 111 to 88 and resize the dual-readout classifier from 222 to 176 inputs, preserving the full causal schedule and training procedure.

EVIDENCE: The 96-unit full-resolution dual-readout model achieved 87.36% accuracy, leaving a 2.36-point margin; the preceding reduction from 104 to 96 units cost only 0.98 points while substantially reducing MACs, motivating another eight-unit reduction without the temporal subsampling that previously failed.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE