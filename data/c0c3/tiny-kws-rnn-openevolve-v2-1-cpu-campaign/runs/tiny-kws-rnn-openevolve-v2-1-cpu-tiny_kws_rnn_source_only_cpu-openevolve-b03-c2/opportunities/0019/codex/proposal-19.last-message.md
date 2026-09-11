MECHANISM: Nested trailing-frame causal subsampling

HYPOTHESIS: A 110-unit dual-readout GRU using 24 nested frames will retain at least 85% validation accuracy while reducing recurrent MACs and executed steps by 4% versus the qualified 25-step design.

INTENDED_EDIT: Adopt the qualified 110-unit width and additionally omit frame 27 from the 25-step schedule, excluding frames 1 and 27–30 for 32-frame inputs.

EVIDENCE: The qualified 110-unit, 25-step design achieved 86.13% accuracy; its preceding 26-to-25-step reduction lost only 0.49 percentage points, leaving a 1.13-point margin for one further isolated trailing-frame removal.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 111, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(222, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 111, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 111, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 110, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(220, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 110, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 110, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if available_frames >= 5:
            return [
                frame
                for frame in schedule
                if frame
                not in (1, available_frames - 3, available_frames - 2)
            ]
=======
        if available_frames >= 6:
            return [
                frame
                for frame in schedule
                if frame
                not in (
                    1,
                    available_frames - 5,
                    available_frames - 4,
                    available_frames - 3,
                    available_frames - 2,
                )
            ]
>>>>>>> REPLACE