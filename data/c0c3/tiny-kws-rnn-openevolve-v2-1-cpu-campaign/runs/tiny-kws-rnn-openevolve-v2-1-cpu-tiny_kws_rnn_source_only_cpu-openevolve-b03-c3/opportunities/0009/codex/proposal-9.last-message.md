MECHANISM: Training-only temporal translation regularization at the recurrent-width boundary

HYPOTHESIS: A 97-unit full-frame GRU trained with random ±2-frame temporal shifts will recover at least 85% validation accuracy while reducing total inference MACs below the qualified 98-unit model’s 905,406,320 MACs.

INTENDED_EDIT: Reduce the GRU and recurrent state width from 98 to 97, and add random zero-padded temporal shifts during training while retaining all 32 recurrent steps at inference.

EVIDENCE: The 97-unit model missed the target by only 0.58 percentage points while achieving lower cross-entropy than the qualified 98-unit model; unlike the failed 24- and 16-frame schedules, temporal translation augmentation preserves full frame density and adds no inference cost.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)
=======
        self.gru = nn.GRU(20, 97, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(97, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 97, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 97, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
    del step, total_steps
    if torch.rand(()) < 0.8:
        frames = frames + 0.025 * torch.randn_like(frames)
    return frames, labels
=======
    del step, total_steps
    if torch.rand(()) < 0.8:
        frames = frames + 0.025 * torch.randn_like(frames)
    if torch.rand(()) < 0.8:
        shift = int(torch.randint(-2, 3, ()).item())
        if shift > 0:
            frames = F.pad(frames[:, :-shift, :], (0, 0, shift, 0))
        elif shift < 0:
            frames = F.pad(frames[:, -shift:, :], (0, 0, 0, -shift))
    return frames, labels
>>>>>>> REPLACE