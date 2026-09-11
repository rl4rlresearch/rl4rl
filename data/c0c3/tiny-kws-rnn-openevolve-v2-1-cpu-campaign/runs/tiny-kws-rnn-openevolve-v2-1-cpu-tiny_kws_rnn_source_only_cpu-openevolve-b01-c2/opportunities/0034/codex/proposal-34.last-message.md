MECHANISM: Gate-selective spectral pruning

HYPOTHESIS: Removing the lowest mel band only from the GRU reset-gate input will retain at least 85% accuracy while reducing total inference MACs from 294,426,900 to approximately 293,204,400.

INTENDED_EDIT: Replace the fused GRU with an equivalent Linear-based GRU whose update and candidate gates retain all 20 bands while its reset gate uses 19; also adopt the qualified bias-free seven-logit head.

EVIDENCE: Removing the lowest band from every GRU gate narrowly missed at 84.42%; retaining that band in the update and candidate paths tests a substantially smaller structural reduction, while the bias-free head already qualified at 85.52%.

<<<<<<< SEARCH
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(3 * self.hidden_size, 7)
=======
        self.reset_input = nn.Linear(19, self.hidden_size)
        self.other_input = nn.Linear(20, 2 * self.hidden_size)
        self.hidden_gates = nn.Linear(
            self.hidden_size, 3 * self.hidden_size
        )
        bound = 1.0 / math.sqrt(self.hidden_size)
        for module in (
            self.reset_input,
            self.other_input,
            self.hidden_gates,
        ):
            nn.init.uniform_(module.weight, -bound, bound)
            nn.init.uniform_(module.bias, -bound, bound)
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
        output = output[:, 0, :]
        return (
            hidden.transpose(0, 1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )
=======
        hidden, summary, maximum, count = state
        previous = hidden[:, 0, :]
        normalized = self.input_norm(frame)
        reset_input = self.reset_input(normalized[:, 1:])
        update_input, candidate_input = self.other_input(
            normalized
        ).chunk(2, dim=1)
        reset_hidden, update_hidden, candidate_hidden = (
            self.hidden_gates(previous).chunk(3, dim=1)
        )
        reset = torch.sigmoid(reset_input + reset_hidden)
        update = torch.sigmoid(update_input + update_hidden)
        candidate = torch.tanh(
            candidate_input + reset * candidate_hidden
        )
        output = (1.0 - update) * candidate + update * previous
        return (
            output.unsqueeze(1),
            summary + output,
            torch.maximum(maximum, output),
            count + 1.0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        hidden, summary, maximum, count = state
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
        return (
            hidden.transpose(0, 1),
            summary + outputs.sum(dim=1),
            torch.maximum(maximum, outputs.amax(dim=1)),
            count + frames.shape[1],
        )
=======
        for index in range(frames.shape[1]):
            state = self.recurrent_step(frames[:, index, :], state)
        return state
>>>>>>> REPLACE