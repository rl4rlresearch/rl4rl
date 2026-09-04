MECHANISM: Width–time exchange with dual-timescale readout

HYPOTHESIS: An 86-unit dual-readout GRU over the most recent 28 frames will achieve at least 85% validation accuracy while reducing predicted total inference MACs below the qualified 88-unit, 27-frame model’s 628,554,080 to approximately 625,202,800.

INTENDED_EDIT: Reduce GRU width from 88 to 86, resize its state and classifier, and expand the trailing window from 27 qualified frames to 28 frames.

EVIDENCE: The 88-unit model achieved 85.89% over 28 frames versus 85.03% over 27; additionally, exchanging two units for one frame previously improved accuracy when moving from the 98-unit, 29-frame model to the 96-unit, 30-frame model.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(176, 8)
=======
        self.gru = nn.GRU(20, 86, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(172, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
=======
        hidden = torch.zeros(batch_size, 1, 86, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 86, device=device, dtype=dtype)
>>>>>>> REPLACE

<<<<<<< SEARCH
        start = max(available_frames - 29, 0)
=======
        start = max(available_frames - 28, 0)
>>>>>>> REPLACE