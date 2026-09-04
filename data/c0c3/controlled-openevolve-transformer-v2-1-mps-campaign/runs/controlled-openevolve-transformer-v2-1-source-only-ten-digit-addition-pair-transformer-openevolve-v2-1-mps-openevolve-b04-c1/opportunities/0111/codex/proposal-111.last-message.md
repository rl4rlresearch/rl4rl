MECHANISM: Pairwise GLU gate-intercept sharing

HYPOTHESIS: Tying only the sixth and seventh GLU gate biases will reduce the verified model from 845 to 844 parameters while retaining at least 99% accuracy, because all seven gate weight vectors and all seven value/output directions remain independently learned.

INTENDED_EDIT: Store thirteen rather than fourteen `fc1` biases and reuse the final stored gate bias for the seventh gate, while preserving the original constructor random draw.

EVIDENCE: The 845-parameter seven-direction GLU reached 99.84%; the more aggressive 837-parameter consensus-gate experiment timed out rather than demonstrating an accuracy failure, so sharing one scalar gate intercept is a conservative test of whether full gate-specific biasing is necessary.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.empty(out_features))
=======
        self.bias = nn.Parameter(torch.empty(out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            self.bias.copy_(conceptual_bias)
=======
        with torch.no_grad():
            self.bias.copy_(conceptual_bias[:-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        return F.linear(x, weight, self.bias)


class OneRowInputGaugeLinear(nn.Module):
=======
        bias = torch.cat((self.bias, self.bias[-1:]))
        return F.linear(x, weight, bias)


class OneRowInputGaugeLinear(nn.Module):
>>>>>>> REPLACE