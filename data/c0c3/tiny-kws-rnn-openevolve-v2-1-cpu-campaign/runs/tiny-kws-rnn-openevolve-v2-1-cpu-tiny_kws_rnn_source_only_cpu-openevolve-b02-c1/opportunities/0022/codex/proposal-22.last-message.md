MECHANISM: Learned low-rank mel input projection

HYPOTHESIS: Compressing each normalized 20-band frame to 16 learned features while preserving the verified 60+59 hidden capacity and 29-step schedule will retain at least 85% validation accuracy and reduce total inference MACs below 646 million.

INTENDED_EDIT: Add a shared bias-free 20-to-16 linear projection and change both GRUs to consume its output, leaving recurrent width, pooling, classifier, training, and frame schedule unchanged.

EVIDENCE: The 60+59 model over frames 3–31 achieved 85.03%, while reducing hidden width to 59+59 failed at 84.29%; this motivates reducing input-side matrix rank without removing recurrent state capacity or additional temporal evidence.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru_a = nn.GRU(20, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(20, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.input_projection = nn.Linear(20, 16, bias=False)
        self.gru_a = nn.GRU(16, 60, num_layers=1, batch_first=True)
        self.gru_b = nn.GRU(16, 59, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
        normalized = self.input_norm(frame).unsqueeze(1)
=======
        normalized = self.input_projection(self.input_norm(frame)).unsqueeze(1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        normalized = self.input_norm(frames)
=======
        normalized = self.input_projection(self.input_norm(frames))
>>>>>>> REPLACE