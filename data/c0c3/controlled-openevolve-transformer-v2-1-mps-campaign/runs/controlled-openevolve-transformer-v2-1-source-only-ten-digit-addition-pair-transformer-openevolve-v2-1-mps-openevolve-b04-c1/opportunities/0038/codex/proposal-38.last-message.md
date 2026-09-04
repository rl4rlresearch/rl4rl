MECHANISM: Distributed residual-stream shift quotient

HYPOTHESIS: Constraining one attention-output projection column, while retaining the two successful `fc2` quotients, will achieve at least 99% accuracy with 1602 parameters because its omitted all-ones component is erased by subsequent LayerNorms without further perturbing MLP-channel optimization.

INTENDED_EDIT: Generalize `OneColumnGaugeLinear` to accept a configurable number of quotient columns and apply a single-column quotient to the attention output projection.

EVIDENCE: Two quotiented `fc2` columns achieved 99.98% accuracy at 1603 parameters, while adding a third collapsed accuracy to 28.45%; distributing the next exact residual-shift quotient to the attention branch tests whether the failure was specific to accumulating constraints in the MLP.

<<<<<<< SEARCH
class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.rest = nn.Linear(in_features - 2, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(2, out_features - 1))
=======
class OneColumnGaugeLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, gauge_columns: int = 2):
        super().__init__()
        self.rest = nn.Linear(in_features - gauge_columns, out_features, bias=False)
        self.first_coordinates = nn.Parameter(torch.empty(gauge_columns, out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model, bias=False)
=======
        self.proj = OneColumnGaugeLinear(d_model, d_model, gauge_columns=1)
>>>>>>> REPLACE