MECHANISM: Nested temporal pruning with boundary preservation

HYPOTHESIS: A 65-unit GRU using 27 frames obtained by removing only the second frame from the verified 28-frame schedule will achieve at least 85% accuracy while matching the 65-unit/27-step model’s 366,004,275 MAC cost.

INTENDED_EDIT: Restore the 65-unit GRU and replace the rephased 27-frame schedule with a 27-frame subset of the successful 28-frame schedule that retains both utterance endpoints.

EVIDENCE: The 65-unit model achieved 85.77% at 28 steps but narrowly missed at 27 steps with 84.79%; the prior 27-step formula moved many sampling locations, so pruning one early, densely sampled frame from the successful schedule isolates frame count from schedule rephasing.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 66, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(198, 8)
=======
        self.gru = nn.GRU(20, 65, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(195, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 66, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 66, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 66), -1.0, device=device, dtype=dtype
        )
=======
        hidden = torch.zeros(batch_size, 1, 65, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 65, device=device, dtype=dtype)
        running_max = torch.full(
            (batch_size, 65), -1.0, device=device, dtype=dtype
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(27, available_frames)
        return [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
        ]
=======
    def frame_schedule(self, available_frames: int) -> list[int]:
        target_steps = min(28, available_frames)
        schedule = [
            step * (available_frames - 1) // (target_steps - 1)
            for step in range(target_steps)
        ]
        if len(schedule) == 28:
            del schedule[1]
        return schedule
>>>>>>> REPLACE