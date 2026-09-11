MECHANISM: More-update training for a width-trimmed strided GRU

HYPOTHESIS: Halving the training batch size will roughly double optimizer updates over the same 50,000 examples, lifting the 124-unit, 16-step model’s prior 84.54% accuracy above 85% while retaining its 699,335,200-MAC inference cost—about 5.7% below the qualified 128-unit design.

INTENDED_EDIT: Use the proven even-index 16-frame schedule, reduce GRU width to 124, and train with batches of 64 for more optimization steps.

EVIDENCE: The 124-unit model missed qualification by only 0.46 percentage points with 397 optimizer steps, while the otherwise identical 128-unit 16-step model reached 85.153%; this suggests testing whether additional optimization can recover the small accuracy deficit without increasing inference cost.

<<<<<<< SEARCH
BATCH_SIZE = 128
=======
BATCH_SIZE = 64
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 128, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(128, 8)
=======
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 128, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 128, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(available_frames))
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        return list(range(0, available_frames, 2))
>>>>>>> REPLACE