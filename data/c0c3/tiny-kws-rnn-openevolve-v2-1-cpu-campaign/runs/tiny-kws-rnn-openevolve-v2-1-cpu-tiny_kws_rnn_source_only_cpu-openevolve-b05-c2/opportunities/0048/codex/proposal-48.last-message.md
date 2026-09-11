MECHANISM: One-unit recurrent-width contraction at the qualified temporal frontier

HYPOTHESIS: A 63-unit signed-extrema GRU on the qualified 21-frame schedule will retain at least 85% validation accuracy while reducing total inference MACs from 277,699,840 to approximately 270,126,045.

INTENDED_EDIT: Restore the qualified 21-step schedule and structurally reduce the GRU width from 64 to 63 units, including its state summaries and classifier input.

EVIDENCE: The 64-unit 21-step design achieved 85.40% accuracy, while several 20-step variants failed; this identifies temporal execution as the current boundary and motivates the smaller, orthogonal 1/64 recurrent-width reduction.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 63, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(252, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 64, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 64, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 63, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        maximum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
        minimum = torch.zeros(batch_size, 63, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            schedule = [index for index in schedule if index not in skipped]
=======
            skipped = {
                available_frames // 8,
                available_frames // 4,
                3 * available_frames // 8,
                available_frames // 2,
                5 * available_frames // 8,
                3 * available_frames // 4,
                7 * available_frames // 8,
            }
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE