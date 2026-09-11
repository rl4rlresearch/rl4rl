MECHANISM: Affine-free normalization at the qualified temporal-width frontier

HYPOTHESIS: The qualified 63-unit, 21-step GRU will retain at least 85% validation accuracy with fixed LayerNorm affine values, preserving 270,126,045 inference MACs while reducing learned parameters from 18,129 to 18,089.

INTENDED_EDIT: Restore the qualified 63-unit GRU and 21-frame schedule, then disable the input LayerNorm’s representationally redundant learned scale and offset.

EVIDENCE: The 63-unit, 21-step design qualified at 85.03%, while recurrent-width, spectral, temporal, and readout contractions failed; LayerNorm starts with the same fixed scale and offset proposed here, and its affine transformation can be absorbed by the following GRU input affine maps.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 64, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 8)
=======
        self.input_norm = nn.LayerNorm(20, elementwise_affine=False)
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
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(15 * available_frames // 16)
            schedule = [index for index in schedule if index not in skipped]
=======
            if available_frames > 16:
                skipped.add(available_frames // 16)
                skipped.add(3 * available_frames // 16)
                skipped.add(15 * available_frames // 16)
            if available_frames >= 32:
                skipped.add(available_frames // 32)
            schedule = [index for index in schedule if index not in skipped]
>>>>>>> REPLACE