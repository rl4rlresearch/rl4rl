MECHANISM: Post-mixing attention offset quotient

HYPOTHESIS: Removing all eight learned value-bias coordinates will reduce the model from 1,528 to 1,520 parameters while retaining at least 99% accuracy, because softmax-normalized attention turns value bias into a position-independent offset that the trainable output-projection bias can represent directly.

INTENDED_EDIT: Challenge the assumption that each attention head needs a learned value offset: retain all head-specific query, key, and value weights and the query bias, but fix value bias to zero and represent learned constant attention offsets only after head mixing.

EVIDENCE: The 1,528-parameter design achieved 100% accuracy, whereas sharing key weights collapsed accuracy to 21.68%, indicating that head-specific addressing is load-bearing. This patch leaves that addressing untouched and removes an exact redundancy: with zero attention dropout, every attention row sums to one, so the value bias always contributes only through the existing mean-free projection bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep constructor RNG consumption, then remove only the softmax-null
        # key bias while retaining the successful full value bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(
            torch.cat((full_bias[:d_model], full_bias[2 * d_model :])).clone()
        )
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the constructor draw, but retain only query bias. Softmax
        # normalization makes value bias a position-common offset already
        # represented by the downstream projection bias.
        full_bias = self.qkv.bias.detach()
        self.qkv.bias = nn.Parameter(full_bias[:d_model].clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias[:d_model]
        v_bias = self.qkv.bias[d_model:]
        full_bias = torch.cat((q_bias, q_bias.new_zeros(d_model), v_bias))
        qkv = F.linear(x, self.qkv.weight) + full_bias
        q, k, v = qkv.chunk(3, dim=-1)
=======
        qkv = F.linear(x, self.qkv.weight)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.qkv.bias
>>>>>>> REPLACE