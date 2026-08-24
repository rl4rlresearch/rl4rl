# Architecture Search Trajectories: 8-, 12-, and 20-Iteration Sweep

This file contains the complete proposal-level results for the 24-run process-intervention sweep: two research frameworks × four memory/deliberation conditions × three search horizons.

## What was measured

- **Runs:** 24
- **Generated proposals:** 320 (seed architectures are shown separately at iteration 0 and are not counted as proposals)
- **Primary trajectory:** the smallest parameter count among candidates that executed successfully, passed the transformer-validity check, and remained eligible as a parent after each proposal
- **RD0:** sequential memory + neutral review
- **RD1:** sequential memory + assumption challenge
- **RD2:** portfolio memory + neutral review
- **RD3:** portfolio memory + assumption challenge

> Important interpretation: validity here is structural/execution validity, not task competence. Public accuracy is reported independently. A tiny zero-accuracy architecture can therefore become the parameter-count incumbent; this is evidence of objective collapse, not a successful AdderBoard model.

> Data provenance: OpenEvolve h8/RD0 iteration 6 is present in the authoritative terminal-outcome ledger and checkpoint, but its asynchronous `evolution_trace.jsonl` event was omitted. This report reconstructs that proposal from `checkpoint_6`; no metrics or architecture fields were imputed.

## Run summary

| Horizon | Framework | Condition | Valid | Unique | Improvements | Final best params | Accuracy of final incumbent | Run ID |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 8 | AutoResearch | RD0 | 6/8 | 8 | 6 | 46 | 0.0000 | `ar-size-h8-20260822-b000-rd0-s1` |
| 8 | AutoResearch | RD1 | 7/8 | 8 | 5 | 3,440 | 0.0000 | `ar-size-h8-20260822-b000-rd1-s1` |
| 8 | AutoResearch | RD2 | 8/8 | 8 | 6 | 2,004 | 0.0000 | `ar-size-h8-20260822-b000-rd2-s1` |
| 8 | AutoResearch | RD3 | 8/8 | 8 | 6 | 756 | 0.0000 | `ar-size-h8-20260822-b000-rd3-s1` |
| 8 | OpenEvolve | RD0 | 4/8 | 4 | 1 | 5,520 | 0.0000 | `oe-size-h8-20260822-b000-rd0-s1` |
| 8 | OpenEvolve | RD1 | 7/8 | 7 | 3 | 2,384 | 0.0000 | `oe-size-h8-20260822-b000-rd1-s1` |
| 8 | OpenEvolve | RD2 | 5/8 | 5 | 3 | 1,888 | 0.0000 | `oe-size-h8-20260822-b000-rd2-s1` |
| 8 | OpenEvolve | RD3 | 7/8 | 7 | 3 | 34 | 0.0000 | `oe-size-h8-20260822-b000-rd3-s1-r2` |
| 12 | AutoResearch | RD0 | 12/12 | 12 | 10 | 234 | 0.0000 | `ar-size-h12-20260822-b000-rd0-s1` |
| 12 | AutoResearch | RD1 | 12/12 | 12 | 12 | 756 | 0.0000 | `ar-size-h12-20260822-b000-rd1-s1` |
| 12 | AutoResearch | RD2 | 11/12 | 12 | 6 | 19 | 0.0000 | `ar-size-h12-20260822-b000-rd2-s1` |
| 12 | AutoResearch | RD3 | 12/12 | 12 | 10 | 376 | 0.0000 | `ar-size-h12-20260822-b000-rd3-s1-r5` |
| 12 | OpenEvolve | RD0 | 8/12 | 8 | 2 | 46 | 0.0000 | `oe-size-h12-20260822-b000-rd0-s1` |
| 12 | OpenEvolve | RD1 | 9/12 | 9 | 4 | 2,384 | 0.0000 | `oe-size-h12-20260822-b000-rd1-s1` |
| 12 | OpenEvolve | RD2 | 10/12 | 10 | 2 | 2,384 | 0.0000 | `oe-size-h12-20260822-b000-rd2-s1` |
| 12 | OpenEvolve | RD3 | 11/12 | 11 | 3 | 1,888 | 0.0000 | `oe-size-h12-20260822-b000-rd3-s1` |
| 20 | AutoResearch | RD0 | 15/20 | 20 | 10 | 19 | 0.0000 | `ar-size-h20-20260822-b000-rd0-s1` |
| 20 | AutoResearch | RD1 | 18/20 | 20 | 16 | 19 | 0.0000 | `ar-size-h20-20260822-b000-rd1-s1` |
| 20 | AutoResearch | RD2 | 9/20 | 20 | 8 | 46 | 0.0000 | `ar-size-h20-20260822-b000-rd2-s1` |
| 20 | AutoResearch | RD3 | 20/20 | 20 | 17 | 376 | 0.0000 | `ar-size-h20-20260822-b000-rd3-s1` |
| 20 | OpenEvolve | RD0 | 17/20 | 17 | 5 | 2,384 | 0.0000 | `oe-size-h20-20260822-b000-rd0-s1-r3` |
| 20 | OpenEvolve | RD1 | 13/20 | 13 | 2 | 2,384 | 0.0000 | `oe-size-h20-20260822-b000-rd1-s1-r3` |
| 20 | OpenEvolve | RD2 | 15/20 | 15 | 4 | 2,384 | 0.0000 | `oe-size-h20-20260822-b000-rd2-s1-r2` |
| 20 | OpenEvolve | RD3 | 16/20 | 16 | 5 | 1,328 | 0.0000 | `oe-size-h20-20260822-b000-rd3-s1` |

## Complete trajectories

Each table includes the shared seed at iteration 0 followed by every generated proposal. `Best valid` is a running minimum and is carried forward when a proposal is invalid.

### AutoResearch · h8 · RD0

- Run: `ar-size-h8-20260822-b000-rd0-s1`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | two-channel causal attention-only decoder with two scalar heads, parameter-free RMS normalization, sinusoidal positions, and tied readout | A two-channel bottleneck retaining separate sinusoidal phase channels and two causal attention heads should preserve the zero public eligibility floor while sharply reducing embedding and attention parameters. |  |
| 1 | `d7b28f0895ed` | 5,520 | 5,520 | yes | yes | 0.0000 | two-layer pre-normalized causal decoder with sinusoidal positions | Replacing learned positions with parameter-free sinusoidal positions should provide stable relative geometry for operand-digit alignment while reducing parameter count. |  |
| 2 | `3fd82dd89139` | 3,952 | 3,952 | yes | yes | 0.0000 | two-layer pre-normalized causal decoder with one lower feed-forward sublayer and sinusoidal positions | A single feed-forward sublayer can compute local digit features before the second attention propagates carry information, making the upper feed-forward redundant. |  |
| 3 | `660b4341e925` | 2,384 | 2,384 | yes | yes | 0.0000 | two-layer pre-normalized causal attention-only decoder with sinusoidal positions | Two sequential causal-attention stages can perform digit alignment and carry propagation without a feed-forward sublayer, reducing parameters while retaining the parent's public floor. |  |
| 4 | `94bac2692ad5` | 1,328 | 1,328 | yes | yes | 0.0000 | single-layer pre-normalized causal attention-only decoder with sinusoidal positions | One position-conditioned causal-attention stage can preserve the incumbent's public floor, making the second attention stage redundant. |  |
| 5 | `40693f6ea531` | 1,264 | 1,264 | yes | yes | 0.0000 | single-layer pre-normalized causal attention-only decoder with parameter-free RMS normalization and sinusoidal positions | Parameter-free RMS normalization should preserve stable attention scaling while eliminating affine normalization parameters that have no demonstrated public benefit. |  |
| 6 | `5318fe38f381` | 46 | 46 | yes | yes | 0.0000 | two-channel causal attention-only decoder with two scalar heads, parameter-free RMS normalization, sinusoidal positions, and tied readout | A two-channel bottleneck retaining separate sinusoidal phase channels and two causal attention heads should preserve the zero public eligibility floor while sharply reducing embedding and attention parameters. |  |
| 7 | `628aecc7d69b` | 19 | 46 | no | yes | 0.0000 | single-channel causal attention-only decoder with one scalar head, parameter-free RMS normalization, sinusoidal positions, and tied readout | A scalar hidden stream with one causal attention head should retain the zero public eligibility floor while reducing both embedding and attention parameters below the two-channel parent. | runtime_transformer_validity |
| 8 | `6597d8edb570` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |

### AutoResearch · h8 · RD1

- Run: `ar-size-h8-20260822-b000-rd1-s1`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Staged Attention-Only Decoder | Removing the early gated pointwise sublayer will preserve positional retrieval geometry for the later attention stages while reducing parameters without losing public eligibility. |  |
| 1 | `321d2405a440` | 6,080 | 6,080 | yes | yes | 0.0000 | Four-Head Alignment Decoder | Four attention heads can specialize into aligned operand retrieval, structural anchoring, and prior-output carry channels without widening the model. |  |
| 2 | `d0a6f850bbfd` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Decoder | Sinusoidal positions provide shared offset structure for operand-digit alignment while removing the learned positional parameter table. |  |
| 3 | `e23e0f7615f0` | 5,136 | 5,136 | yes | yes | 0.0000 | Gated Carry Decoder | A narrow SiLU-gated second feed-forward block can represent digit-sum and carry conjunctions more efficiently than the wider additive GELU block. |  |
| 4 | `eb657e3293d5` | 5,008 | 5,008 | yes | yes | 0.0000 | Staged Carry Attention Decoder | Replacing the gated pointwise carry stage with a third causal attention stage will let the model separately retrieve aligned operand digits and integrate generated carry context while using fewer parameters. |  |
| 5 | `b20280835e4d` | 4,240 | 4,240 | yes | yes | 0.0000 | Staged Attention Compact Gate Decoder | A compact SiLU-gated pointwise stage will multiplicatively compose retrieved digit and carry features before the later attention stages, providing conditional arithmetic structure with fewer parameters than the wide GELU stage. |  |
| 6 | `41132ae1bf4e` | 4,800 | 4,240 | yes | yes | 0.0000 | Staged Attention Learned Alignment Decoder | Learned position vectors will make fixed operand columns and reversed-output steps directly distinguishable to compact attention heads, enabling reliable digit alignment without changing the arithmetic or carry-computation stages. |  |
| 7 | `a521b57b3298` | — | 4,240 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 8 | `619d7e1e123a` | 3,440 | 3,440 | yes | yes | 0.0000 | Staged Attention-Only Decoder | Removing the early gated pointwise sublayer will preserve positional retrieval geometry for the later attention stages while reducing parameters without losing public eligibility. |  |

### AutoResearch · h8 · RD2

- Run: `ar-size-h8-20260822-b000-rd2-s1`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Serial Operand Refinement Adder Decoder | Replacing parallel attention and projection fusion with two serial residual attention stages should let the second stage refine aligned operand evidence produced by the first while reducing parameters. |  |
| 1 | `2be9fb415cc2` | 6,080 | 6,080 | yes | yes | 0.0000 | Four-Head AdderBoard Decoder | Four attention heads create separate retrieval channels for aligned operand digits, prior output, and carry context without increasing parameter count. |  |
| 2 | `229fcf006b41` | 6,080 | 6,080 | yes | yes | 0.0000 | Gated Arithmetic AdderBoard Decoder | Width-32 SiLU-gated feed-forwards can form multiplicative digit-and-carry conjunctions more directly than width-48 GELU feed-forwards while preserving the parent parameter count. |  |
| 3 | `693148685ec6` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Adder Decoder | Parameter-free sinusoidal positions will expose stable relative offsets between answer digits and aligned operand digits, improving attention-based retrieval while reducing parameters. |  |
| 4 | `a2f4cf5f802b` | 3,952 | 3,952 | yes | yes | 0.0000 | Retrieval Then Compute Adder Decoder | Removing the first feed-forward sublayer will preserve digit and positional codes through the retrieval stage, leaving the second gated feed-forward to compute digit sums and carry while reducing parameters. |  |
| 5 | `5d9ec43d5135` | 2,896 | 2,896 | yes | yes | 0.0000 | Single Retrieval Compute Adder Decoder | A single causal attention layer can retrieve the aligned operand digits needed at each output position, while the gated feed-forward computes digit sums and carry without interference from a redundant retrieval stage. |  |
| 6 | `5f2b6665b2b5` | 2,384 | 2,384 | yes | yes | 0.0000 | Single Attention GELU Adder Decoder | Replacing multiplicative gated computation with a parameter-efficient GELU feed-forward should provide a smoother digit-and-carry transformation while preserving aligned retrieval and reducing parameters. |  |
| 7 | `f76af933702d` | 2,268 | 2,268 | yes | yes | 0.0000 | Parallel Operand Retrieval Adder Decoder | Two parallel causal-attention streams should retrieve the aligned operands separately before fusion, reducing destructive averaging and enabling a compact GELU block to compute digit and carry transitions with fewer parameters. |  |
| 8 | `f391d1636726` | 2,004 | 2,004 | yes | yes | 0.0000 | Serial Operand Refinement Adder Decoder | Replacing parallel attention and projection fusion with two serial residual attention stages should let the second stage refine aligned operand evidence produced by the first while reducing parameters. |  |

### AutoResearch · h8 · RD3

- Run: `ar-size-h8-20260822-b000-rd3-s1`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Width-12 Single-Attention Addition Decoder | A twelve-dimensional residual stream with four three-dimensional attention heads should retain the parent's causal alignment behavior while eliminating width coordinates not supported by any public arithmetic benefit. |  |
| 1 | `5a04d779a320` | 6,080 | 6,080 | yes | yes | 0.0000 | Four-Head Addition Decoder | Four attention heads at unchanged parameter count will permit separate specialization for operand alignment, output-position tracking, prior-digit context, and carry propagation. |  |
| 2 | `6b74846d06c4` | 6,080 | 6,080 | yes | yes | 0.0000 | Gated Carry Addition Decoder | Parameter-matched SiLU-gated feed-forward blocks will create multiplicative digit-and-carry interactions that ordinary GELU blocks cannot represent as efficiently, improving autoregressive addition without increasing parameter count. |  |
| 3 | `6cf6b175fcaa` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Addition Decoder | Replacing learned positions with sinusoidal positions will expose regular relative offsets between aligned operand and answer digits while eliminating positional parameters. |  |
| 4 | `5862a873dc7a` | 2,384 | 2,384 | yes | yes | 0.0000 | Attention-Only Addition Decoder | Removing both gated feed-forward sublayers will preserve eligibility at substantially lower parameter count if stacked causal attention already supplies enough nonlinear alignment and carry-state transformation. |  |
| 5 | `0d0886a4ae89` | 1,328 | 1,328 | yes | yes | 0.0000 | Single-Attention Addition Decoder | A single causal attention block can preserve the current eligibility floor by directly aligning each answer position with operand digits, making the second attention block unnecessary. |  |
| 6 | `947abc45bcf3` | 1,296 | 1,296 | yes | yes | 0.0000 | Single-Attention RMS-Normalized Addition Decoder | Replacing mean-centering layer normalization with RMS normalization should preserve embedding-coordinate signals while reducing affine normalization parameters and retaining the public eligibility floor. |  |
| 7 | `47a1ebb5299a` | 1,264 | 1,264 | yes | yes | 0.0000 | Single-Attention Non-Affine RMS Addition Decoder | Non-affine RMS normalization should preserve stable attention scaling while removing learned normalization parameters that have shown no public arithmetic benefit. |  |
| 8 | `adc950847608` | 756 | 756 | yes | yes | 0.0000 | Width-12 Single-Attention Addition Decoder | A twelve-dimensional residual stream with four three-dimensional attention heads should retain the parent's causal alignment behavior while eliminating width coordinates not supported by any public arithmetic benefit. |  |

### OpenEvolve · h8 · RD0

- Run: `oe-size-h8-20260822-b000-rd0-s1`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions reduce parameters and provide stable digit-place structure for autoregressive addition. |  |
| 2 | `984334851f70` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Depth-Mix AdderBoard Decoder | A fixed shallow-to-deep mixture preserves token-position cues while causal attention computes carry dependencies. |  |
| 3 | `03cf58f8-7f9` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions reduce parameters and provide a stable ordinal signal for carry propagation. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `ec63d6ec-11e` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions preserve sequence ordering while removing the learned position table. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 5 | `4d6552f9-028` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide an immediately usable ordering signal while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 6 | `731136311205` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Depth-Mix AdderBoard Decoder | Parameter-free sinusoidal positions provide structured digit-place cues while reducing model size relative to learned absolute positions. |  |
| 7 | `c2c5666f-baf` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide arithmetic order information without requiring learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 8 | `7c0f9305cac9` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Depth Mix | A parameter-free late-depth mixture preserves useful first-block digit features while retaining deeper carry computation. |  |

### OpenEvolve · h8 · RD1

- Run: `oe-size-h8-20260822-b000-rd1-s1`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide a lower-parameter relative-distance bias that can improve addition learning. |  |
| 2 | `67af692109c0` | 6,083 | 5,520 | yes | yes | 0.0000 | Multi-Depth Routed AdderBoard Decoder | Learned softmax mixing of embedding, intermediate, and deep streams preserves local digit cues while retaining causal carry features. |  |
| 3 | `3b241581eafd` | 2,944 | 2,944 | yes | yes | 0.0000 | Attention-Only AdderBoard Decoder | An attention-only residual decoder can preserve eligibility while eliminating feed-forward parameters that showed no public task benefit. |  |
| 4 | `c4dfc56f-05c` | — | 2,944 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal coordinates reduce positional learning burden and parameter count while preserving arithmetic-relevant order. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 5 | `fb9401bee2f6` | 5,520 | 2,944 | yes | yes | 0.0000 | Sinusoidal Multi-Depth Mix Decoder | A parameter-free mixture of embedding-level, first-block, and second-block states preserves digit identity while retaining progressively computed carry features. |  |
| 6 | `5227bc9078c6` | 3,456 | 2,944 | yes | yes | 0.0000 | Direct Shallow AdderBoard Decoder | A single direct causal block can avoid depth-routing dilution while retaining sufficient carry computation and using fewer parameters. |  |
| 7 | `babc7fa2200e` | 2,384 | 2,384 | yes | yes | 0.0000 | Sinusoidal Attention-Only AdderBoard Decoder | Fixed sinusoidal positions can expose stable digit-distance structure while eliminating learned positional parameters. |  |
| 8 | `7f3730207ea2` | 5,520 | 2,384 | yes | yes | 0.0000 | Sinusoidal AdderBoard Shallow-Deep Mix Decoder | A parameter-free shallow-state bypass preserves token-local digit evidence while the deep branch supplies causal carry context. |  |

### OpenEvolve · h8 · RD2

- Run: `oe-size-h8-20260822-b000-rd2-s1`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions preserve causal capacity while reducing parameters and supplying stable digit-place structure. |  |
| 2 | `cce26876-23d` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Sinusoidal positions provide a parameter-free ordinal bias that can improve arithmetic generalization while preserving decoder validity. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 3 | `26cabaf0-a47` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions reduce parameters while providing stable absolute-order cues for carry propagation. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `c9d919db2e03` | 3,424 | 3,424 | yes | yes | 0.0000 | Parallel Fusion AdderBoard Decoder | Equal fixed fusion of parallel causal-context and token-local branches can shorten optimization paths while retaining arithmetic context. |  |
| 5 | `fb25ef07af49` | 5,360 | 3,424 | yes | yes | 0.0000 | Affine-Free RMS AdderBoard Decoder | Affine-free RMS normalization preserves residual magnitudes while removing unnecessary normalization parameters and may improve small-decoder optimization. |  |
| 6 | `047df66c-4b1` | — | 3,424 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions can preserve decoder validity while removing learned positional parameters and providing a stable position signal for carry propagation. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 7 | `f727723b7f81` | 3,952 | 3,424 | yes | yes | 0.0000 | Sinusoidal Fixed-Depth-Mix AdderBoard Decoder | A fixed shallow-deep mix after the second causal attention can retain carry context while eliminating a redundant second feed-forward sublayer. |  |
| 8 | `b9ee85f83ffc` | 1,888 | 1,888 | yes | yes | 0.0000 | Attention-Only Residual AdderBoard Decoder | A direct causal-attention residual avoids dilution by the token-local branch while removing its parameters. |  |

### OpenEvolve · h8 · RD3

- Run: `oe-size-h8-20260822-b000-rd3-s1-r2`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide stable digit alignment while removing learned positional parameters. |  |
| 2 | `9b15cc1db468` | 6,080 | 5,520 | yes | yes | 0.0000 | Gated Carry Decoder | Parameter-matched SiLU-gated feed-forward blocks can represent carry-conditioned digit transformations better than GELU MLPs. |  |
| 3 | `4ff6ece0b64a` | 46 | 46 | yes | yes | 0.0000 | Minimal Rank-2 Residual Attention Adder | A minimal rank-2 residual causal-attention bottleneck can preserve eligibility without feed-forward depth. |  |
| 4 | `401bbe73bcf8` | 6,080 | 46 | yes | yes | 0.0000 | Global Embedding Skip AdderBoard Decoder | A fixed global skip preserves token and position information through the decoder without adding trainable parameters. |  |
| 5 | `f19aeaa5-b63` | — | 46 | no | no | — | Gated Carry Interaction Decoder | SiLU-gated feed-forward blocks can represent digit-and-carry interactions more effectively than additive GELU MLPs at a matched projection budget. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 6 | `8ec8ff15cb11` | 5,520 | 46 | yes | yes | 0.0000 | Sinusoidal Gated Carry Decoder | Fixed sinusoidal positions provide a parameter-free ordinal basis that can make digit alignment easier to learn than unconstrained learned positions. |  |
| 7 | `44589818c018` | 34 | 34 | yes | yes | 0.0000 | Scalar Affine Residual Attention Adder | An unnormalized scalar residual can encode arithmetic state continuously, while class-specific readout biases let tied scalar logits discriminate all tokens. |  |
| 8 | `2fdd2ba5fc7c` | 6,080 | 34 | yes | yes | 0.0000 | Parameter-Matched Gated AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks can represent carry-conditioned transformations more effectively than additive GELU blocks. |  |

### AutoResearch · h12 · RD0

- Run: `ar-size-h12-20260822-b000-rd0-s1`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Six-Channel Three-Head RMS Decoder | Reducing the residual width from twelve to six while retaining three causal attention heads will test whether the zero-scoring parent contains unused channel capacity that can be removed without lowering public eligibility. |  |
| 1 | `3bf8ad4b5e73` | 7,136 | 6,080 | yes | yes | 0.0000 | Attention Refinement Adder | A third attention-only refinement stage will provide an extra causal communication round for combining aligned operand evidence with the preceding generated digit that encodes carry state, improving addition without the parameter cost of another feed-forward block. |  |
| 2 | `0b8e990dd7a1` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Decoder | Fixed sinusoidal positions will provide structured absolute phase cues for operand-column alignment while eliminating learned positional parameters. |  |
| 3 | `f2484d9f8519` | 7,056 | 5,520 | yes | yes | 0.0000 | Sinusoidal Gated Arithmetic Decoder | SILU-gated feed-forward blocks will create multiplicative digit-feature interactions that more directly represent sum and carry conditions than GELU feed-forward blocks. |  |
| 4 | `53e466d4ec58` | 4,668 | 4,668 | yes | yes | 0.0000 | Narrow Three-Stage Carry Decoder | Reallocating capacity from width to a third causal transformation stage will enable an additional alignment-to-digit-to-carry refinement step while reducing total parameters. |  |
| 5 | `7add490a2d23` | 4,560 | 4,560 | yes | yes | 0.0000 | Narrow Three-Stage Gated Carry Decoder | Compact SiLU-gated feed-forward blocks will model multiplicative digit-and-carry interactions more effectively than wider GELU blocks while using fewer parameters. |  |
| 6 | `e4fe2145e310` | 4,476 | 4,476 | yes | yes | 0.0000 | Narrow Three-Stage RMS Carry Decoder | Replacing mean-centering LayerNorm with RMSNorm will preserve magnitude-coded digit and carry features while removing normalization offset parameters. |  |
| 7 | `7838277b34b1` | 3,648 | 3,648 | yes | yes | 0.0000 | Narrow Three-Stage GELU Carry Decoder | Replacing gated feed-forwards with GELU feed-forwards will retain smooth nonlinear digit and carry transformations while eliminating the parameter-heavy multiplicative gating projections. |  |
| 8 | `dfe36542474f` | 1,956 | 1,956 | yes | yes | 0.0000 | Narrow Three-Stage Attention-Only Decoder | Removing all feed-forward sublayers will preserve the parent eligibility level because three causal attention stages can still perform positional digit alignment and carry routing, while substantially reducing parameters. |  |
| 9 | `b00319ab89da` | 1,368 | 1,368 | yes | yes | 0.0000 | Narrow Two-Stage Attention-Only Decoder | Two causal attention stages can retain the current eligibility level by separating positional digit alignment from carry-aware aggregation, making the unsupported third stage removable. |  |
| 10 | `d610b7b143e7` | 780 | 780 | yes | yes | 0.0000 | Narrow Single-Stage Attention-Only Decoder | One causal attention stage may preserve the current eligibility floor because reversed autoregressive targets expose prior output digits as carry context, making the second attention stage unnecessary. |  |
| 11 | `50fe95beb8e1` | 756 | 756 | yes | yes | 0.0000 | Narrow Single-Stage Parameter-Free RMS Decoder | Parameter-free RMS normalization should retain scale stabilization while removing learned gains that are not evidenced as useful in the zero-scoring parent. |  |
| 12 | `f7a4df79a9f5` | 234 | 234 | yes | yes | 0.0000 | Six-Channel Three-Head RMS Decoder | Reducing the residual width from twelve to six while retaining three causal attention heads will test whether the zero-scoring parent contains unused channel capacity that can be removed without lowering public eligibility. |  |

### AutoResearch · h12 · RD1

- Run: `ar-size-h12-20260822-b000-rd1-s1`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Width Twelve Routing Capacity Test | Twelve channels split across two attention heads can preserve operand routing and carry-state capacity while reducing embedding and attention parameters. |  |
| 1 | `cd0181c04f3d` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Test | Parameter-free sinusoidal positions will supply stable digit-offset cues for operand alignment and reduce parameters; improved public accuracy would indicate that learned positional alignment was the bottleneck. |  |
| 2 | `5e1e212636ad` | 4,752 | 4,752 | yes | yes | 0.0000 | Compact Gated Carry Test | Compact SiLU-gated feed-forward blocks will create multiplicative digit-feature interactions that represent sum and carry conditions more efficiently than wider additive GELU blocks. |  |
| 3 | `019ace4da8b8` | 4,496 | 4,496 | yes | yes | 0.0000 | Iterative Attention Carry Test | Replacing token-local gated transformations with four successive causal-attention stages will create iterative digit retrieval and carry routing while using fewer parameters than the two-block gated parent. |  |
| 4 | `33a04b8cf44e` | 4,240 | 4,240 | yes | yes | 0.0000 | Three Route Gated Carry Test | Replacing the fourth attention stage with a gated local transformation will let three routing stages align operand digits while multiplicative features compute digit sums and carry conditions, with fewer parameters. |  |
| 5 | `8565992e5afc` | 3,984 | 3,984 | yes | yes | 0.0000 | Two Route Two Gated Carry Test | Two causal attention stages can retrieve aligned digits and prior carry context; replacing the third attention stage with a second gated transformation should reduce parameters while adding serial multiplicative depth for sum and carry computation. |  |
| 6 | `47715a38baf0` | 2,928 | 2,928 | yes | yes | 0.0000 | One Route Two Gated Carry Test | A single causal attention stage can jointly retrieve aligned operand digits and prior generated carry context, so deleting the unsupported second routing stage while retaining two gated residual transformations should preserve eligibility with fewer parameters. |  |
| 7 | `fb3a82b259d2` | 2,128 | 2,128 | yes | yes | 0.0000 | One Route One Gated Carry Test | A single gated residual transformation can compute output digits and carry updates from attention-routed operands and prior outputs, making the first of the parent's two post-attention gated stages redundant. |  |
| 8 | `1d1ab184ec0a` | 1,328 | 1,328 | yes | yes | 0.0000 | Attention Direct Readout Test | If causal attention already constructs aligned digit and carry features, its residual output should remain equally eligible without the post-attention gated transformation. |  |
| 9 | `38090419f299` | 1,296 | 1,296 | yes | yes | 0.0000 | Fixed Final Normalization Test | Parameter-free final layer normalization should preserve logit-relevant standardization while removing redundant learned calibration before the tied readout. |  |
| 10 | `a2a652e53e21` | 1,264 | 1,264 | yes | yes | 0.0000 | Fixed Route Normalization Test | Parameter-free pre-attention normalization should retain stable attention geometry while allowing attention projections to absorb unnecessary learned scale and offset. |  |
| 11 | `08aa14f07bce` | 994 | 994 | yes | yes | 0.0000 | Width Fourteen Routing Capacity Test | Two seven-channel attention heads should retain sufficient operand-routing capacity while eliminating redundant representation dimensions. |  |
| 12 | `ba683c5e9ac0` | 756 | 756 | yes | yes | 0.0000 | Width Twelve Routing Capacity Test | Twelve channels split across two attention heads can preserve operand routing and carry-state capacity while reducing embedding and attention parameters. |  |

### AutoResearch · h12 · RD2

- Run: `ar-size-h12-20260822-b000-rd2-s1`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Scalar Residual Dual-Position Attention Decoder | A parameter-free residual around causal attention will preserve local token and position features while adding contextual carry information, improving accuracy at unchanged parameter count. |  |
| 1 | `b7271b5f1eb2` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Position Adder Decoder | Replacing learned positions with parameter-free sinusoidal positions will expose stable digit-offset relationships for operand alignment while reducing parameter count without weakening causal carry computation. |  |
| 2 | `0b1d79a1fdf5` | 5,360 | 5,360 | yes | yes | 0.0000 | Fixed-Normalization Sinusoidal Adder Decoder | Removing affine scale and bias from every layer normalization will preserve residual-stream stabilization needed for digit alignment and carry propagation while eliminating normalization parameters that the zero-score parent has not shown to be useful. |  |
| 3 | `962831e80cab` | 2,800 | 2,800 | yes | yes | 0.0000 | Single-Block Sinusoidal Adder Decoder | A single causal attention and feed-forward block can perform digit alignment and local carry computation directly in the autoregressive residual stream, making the second refinement block unnecessary for preserving the zero-score parent eligibility floor. |  |
| 4 | `708bee904909` | 1,264 | 1,264 | yes | yes | 0.0000 | Attention-Only Sinusoidal Adder Decoder | For the current zero-score eligibility floor, causal attention alone can preserve the required token-history dependence, so deleting the feed-forward sublayer should retain eligibility while removing its parameters. |  |
| 5 | `3c337ba2df38` | 46 | 46 | yes | yes | 0.0000 | Width-2 Attention-Only Sinusoidal Adder Decoder | A two-dimensional residual stream with two single-coordinate attention heads can preserve the parent's zero-score eligibility floor while sharply reducing all width-dependent parameters and retaining causal token-history mixing. |  |
| 6 | `b4d23d9f0291` | 19 | 46 | no | yes | 0.0000 | Scalar RMS Attention-Only Adder Decoder | A scalar residual stream with parameter-free RMS normalization will avoid the exact collapse of width-one layer normalization, preserve measurable causal attention, and match the zero-score eligibility floor with fewer parameters. | runtime_transformer_validity |
| 7 | `649c4303102a` | 19 | 19 | yes | yes | 0.0000 | Scalar Attention Residual Sinusoidal Adder Decoder | A normalization-free scalar residual stream will keep causal attention measurable while reducing the model to one embedding coordinate and the minimum one-head attention width. |  |
| 8 | `067a03a9cadf` | 19 | 19 | yes | yes | 0.0000 | Scalar Direct Attention Sinusoidal Adder Decoder | Removing the residual bypass will force the tied logits to depend directly on causal attention, potentially improving accuracy without adding parameters. |  |
| 9 | `02ecf29a4e96` | 19 | 19 | yes | yes | 0.0000 | Scalar Post-Attention Position Adder Decoder | Moving sinusoidal position injection after causal attention will keep scalar attention focused on token content while retaining position-dependent logits, improving accuracy without adding parameters. |  |
| 10 | `c1c99c27e32a` | 19 | 19 | yes | yes | 0.0000 | Scalar Dual-Position Attention Decoder | Adding parameter-free position signals before attention will enable alignment-dependent causal retrieval while retaining post-attention position information for digit-specific logits. |  |
| 11 | `24200059ad7a` | 19 | 19 | yes | yes | 0.0000 | Scalar Residual Dual-Position Attention Decoder | A parameter-free residual around causal attention will preserve local token and position features while adding contextual carry information, improving accuracy at unchanged parameter count. |  |
| 12 | `5736b7305ffd` | 2,800 | 19 | yes | yes | 0.0000 | Multichannel Gated Digit-Compute Decoder | Four causal attention heads can retrieve the two aligned digits and prior-column evidence in separate channels, while a gated feed-forward residual computes the nonlinear digit and carry transformation. |  |

### AutoResearch · h12 · RD3

- Run: `ar-size-h12-20260822-b000-rd3-s1-r5`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | eight-channel attention-only residual model with sinusoidal positions, non-affine RMS normalization, and tied readout | Reducing the residual width from 16 to 8 should preserve the attention-based alignment mechanism while removing channels not required to maintain the current public eligibility floor. |  |
| 1 | `c3c4b9909768` | 6,080 | 6,080 | yes | yes | 0.0000 | Gated Carry Interaction Decoder | At fixed parameter budget, SiLU-gated feed-forwards will model multiplicative digit-and-carry interactions more reliably than GELU feed-forwards. |  |
| 2 | `3540f0ac33f4` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Carry Decoder | Replacing learned absolute positions with parameter-free sinusoidal positions will expose regular digit-column offsets, improving alignment while reducing parameter count. |  |
| 3 | `30043b67609c` | 4,496 | 4,496 | yes | yes | 0.0000 | Sinusoidal GELU Carry Decoder | Replacing gated feed-forwards with parameter-cheaper GELU feed-forwards will preserve nonlinear digit and carry computation while removing unnecessary multiplicative projections that may impede optimization. |  |
| 4 | `4ef65fc19d29` | 3,440 | 3,440 | yes | yes | 0.0000 | two causal attention stages followed by one GELU feed-forward stage | Two causal attention stages can gather aligned operand digits and prior-output carry evidence before one final GELU stage computes the next digit, making the first block's nonlinear sublayer redundant. |  |
| 5 | `ec32fda0c7bd` | 3,408 | 3,408 | yes | yes | 0.0000 | two parallel causal attention branches with fixed mixing and one GELU feed-forward stage | Parallel causal attention branches sharing the same normalized positional input can independently retrieve aligned digits and carry evidence without sequentially distorting absolute-position features. |  |
| 6 | `d6ca0244fbcb` | 3,392 | 3,392 | yes | yes | 0.0000 | parallel causal attention with learned absolute positions and a parameter-compensating feed-forward bottleneck | Learned absolute slot embeddings should expose the fixed operand and answer alignment directly to attention, while a narrower nonlinear stage keeps total parameters below the parent. |  |
| 7 | `0637a04a3a00` | 3,392 | 3,392 | yes | yes | 0.0000 | serial residual alignment and carry attention with learned absolute positions | Serializing the attention stages should let carry retrieval condition on an intermediate aligned-digit representation instead of combining two independent retrievals. |  |
| 8 | `499c72808f2a` | 2,368 | 2,368 | yes | yes | 0.0000 | single residual attention followed by local arithmetic feed-forward | One causal attention can jointly retrieve aligned operands and prior generated digits, making a separate serial carry-attention stage unnecessary for the feed-forward network to compute the next digit. |  |
| 9 | `a53b0d3ada38` | 1,808 | 1,808 | yes | yes | 0.0000 | single residual attention with fixed sinusoidal positions and local arithmetic feed-forward | Fixed sinusoidal positions provide stable digit-offset signals for aligned retrieval while eliminating the learned positional table, preserving the one-attention computation with fewer parameters. |  |
| 10 | `8f9aa1e75b8f` | 1,328 | 1,328 | yes | yes | 0.0000 | attention-only residual model with sinusoidal positions and tied readout | Removing the feed-forward sublayer tests whether causal attention's content-dependent mixing already supplies sufficient nonlinear alignment behavior, allowing the residual stream to preserve eligibility with substantially fewer parameters. |  |
| 11 | `eeb093bcee13` | 1,264 | 1,264 | yes | yes | 0.0000 | attention-only residual model with sinusoidal positions, non-affine RMS normalization, and tied readout | Replacing affine layer normalization with non-affine RMS normalization should preserve residual-stream scale control while removing normalization parameters that have no demonstrated benefit. |  |
| 12 | `cdc520047c43` | 376 | 376 | yes | yes | 0.0000 | eight-channel attention-only residual model with sinusoidal positions, non-affine RMS normalization, and tied readout | Reducing the residual width from 16 to 8 should preserve the attention-based alignment mechanism while removing channels not required to maintain the current public eligibility floor. |  |

### OpenEvolve · h12 · RD0

- Run: `oe-size-h12-20260822-b000-rd0-s1`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `cb88d106a19f` | 6,082 | 6,080 | yes | yes | 0.0000 | Adaptive Depth Mix AdderBoard Decoder | A learned softmax mixture of shallow and deep block outputs preserves local digit evidence while retaining deeper carry computation. |  |
| 2 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Parameter-free sinusoidal positions provide structured digit-location cues while reducing parameter count. |  |
| 3 | `787ff5a5-f12` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions preserve causal sequence information while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `c07f566e-977` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions reduce parameters while providing stable digit-place signals for autoregressive addition. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 5 | `0f0cc328-e03` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions preserve digit-offset structure while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 6 | `8ec8ff15cb11` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal SiLU-Gated AdderBoard Decoder | SiLU-gated feed-forward blocks can model multiplicative carry interactions more effectively than GELU blocks at a comparable projection budget. |  |
| 7 | `4661a87a6515` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Depth Mix AdderBoard Decoder | A parameter-free fixed depth mixture preserves shallow digit features while emphasizing deeper carry features without adaptive routing collapse. |  |
| 8 | `a36e2bcb9a6c` | 46 | 46 | yes | yes | 0.0000 | Compact Residual Attention AdderBoard Decoder | A single narrow residual attention path can preserve eligibility while eliminating parameters from depth routing, feed-forward layers, learned positions, and normalization. |  |
| 9 | `d8ec010b58de` | 6,080 | 46 | yes | yes | 0.0000 | Fixed Depth Mix AdderBoard Decoder | A fixed equal mix of shallow and deep features avoids adaptive-router instability while preserving both token identity and carry computation. |  |
| 10 | `079802f9-407` | — | 46 | no | no | — | Sinusoidal Gated AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks can model multiplicative digit and carry interactions better than GELU blocks. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 11 | `fa546f4fc2b9` | 4,512 | 46 | yes | yes | 0.0000 | Attention-Only Refinement AdderBoard Decoder | An attention-only second block can preserve causal carry refinement while removing a redundant late feed-forward sublayer and its parameters. |  |
| 12 | `2d0a0324b705` | 46 | 46 | yes | yes | 0.0000 | Fixed-Mix Compact Attention AdderBoard Decoder | A unit-sum fixed mixture can stabilize the narrow attention residual stream without adding trainable parameters. |  |

### OpenEvolve · h12 · RD1

- Run: `oe-size-h12-20260822-b000-rd1-s1`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions can preserve ordering while removing unnecessary learned position parameters. |  |
| 2 | `ca6621ff-6f8` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Deterministic sinusoidal positions preserve ordering while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 3 | `b2e1da81-e8e` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide stable digit-offset structure while reducing parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `95cf01e68f60` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Multidepth AdderBoard Decoder | A parameter-free mixture of positional, intermediate, and deep features preserves exact digit cues that deepest-only readout may erase. |  |
| 5 | `bb14aa0c4899` | 3,952 | 3,952 | yes | yes | 0.0000 | Attention-Only Refinement Adder Decoder | An attention-only second refinement block can propagate carries without a redundant second feed-forward transformation. |  |
| 6 | `bf0a6cd6-4a5` | — | 3,952 | no | no | — | Sinusoidal AdderBoard Decoder | Sinusoidal positions may improve arithmetic alignment while removing learned position parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 7 | `22d1a7729fa3` | 5,520 | 3,952 | yes | yes | 0.0000 | Sinusoidal Multidepth Mix Decoder | A parameter-free average of shallow and deep representations can preserve digit identity while retaining transformed carry information. |  |
| 8 | `0e6ce8e8a107` | 6,080 | 3,952 | yes | yes | 0.0000 | Gated Carry Multidepth AdderBoard Decoder | Parameter-matched SiLU gating can represent conditional carry transformations more effectively than additive GELU feed-forward blocks. |  |
| 9 | `4bd41d6fefe2` | 2,896 | 2,896 | yes | yes | 0.0000 | Single-Attention Carry Adder Decoder | One causal attention stage plus a nonlinear residual transformation is sufficient for carry propagation, making a second attention refinement stage redundant. |  |
| 10 | `fb25ef07af49` | 5,360 | 2,896 | yes | yes | 0.0000 | Sinusoidal RMS AdderBoard Decoder | Non-affine RMS normalization can preserve stable causal computation while removing unnecessary centering and normalization parameters. |  |
| 11 | `19933b6dca67` | 2,384 | 2,384 | yes | yes | 0.0000 | Attention-Only Multidepth Decoder | Removing positionwise feed-forward branches preserves token identity and reduces optimization burden while causal attention supplies the carry information needed for addition. |  |
| 12 | `9b15cc1db468` | 6,080 | 2,384 | yes | yes | 0.0000 | Direct Deep Gated Carry AdderBoard Decoder | Removing fixed multidepth averaging will prevent shallow representations from diluting the deepest carry-sensitive state. |  |

### OpenEvolve · h12 · RD2

- Run: `oe-size-h12-20260822-b000-rd2-s1`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `9b15cc1db468` | 6,080 | 6,080 | yes | yes | 0.0000 | Gated Carry AdderBoard Decoder | SiLU-gated feed-forwards can represent conditional digit and carry transformations more effectively than additive GELU feed-forwards. |  |
| 2 | `0f75cbb6-093` | — | 6,080 | no | no | — | Gated SiLU AdderBoard Decoder | Parameter-matched SiLU gating improves multiplicative carry-feature learning. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 3 | `30feb319ca36` | 6,080 | 6,080 | yes | yes | 0.0000 | AdderBoard Global Token Bypass Experiment | A parameter-free positional bypass preserves digit identity while the causal stack computes carry context. |  |
| 4 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Sinusoidal positions provide a structured ordinal signal while removing the learned absolute-position parameter table. |  |
| 5 | `6d3e7c9b-e37` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Parameter-free sinusoidal positions preserve absolute digit alignment while reducing learned parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 6 | `8ec8ff15cb11` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Gated Carry AdderBoard Decoder | Fixed sinusoidal positions provide a structured digit-place signal while removing learned positional parameters. |  |
| 7 | `95cf01e68f60` | 6,080 | 5,520 | yes | yes | 0.0000 | AdderBoard Multi-Depth Fixed Mix Experiment | A parameter-free mixture of token, first-block, and second-block features exposes complementary digit and carry representations to the readout. |  |
| 8 | `babc7fa2200e` | 2,384 | 2,384 | yes | yes | 0.0000 | Sinusoidal Attention-Only AdderBoard Decoder | Removing tokenwise feed-forward branches focuses capacity on causal digit routing and carry propagation while reducing parameters. |  |
| 9 | `594bb7ab9b39` | 6,083 | 2,384 | yes | yes | 0.0000 | Adaptive Depth-Mix AdderBoard Decoder | A learned softmax mixture of embedding-level, first-block, and second-block states can preserve digit identity while adaptively selecting the depth needed for carry computation. |  |
| 10 | `fe03f0042572` | 5,360 | 2,384 | yes | yes | 0.0000 | Sinusoidal Parameter-Free RMS Carry Decoder | Parameter-free RMS normalization can preserve stable residual scaling while removing unnecessary affine normalization parameters. |  |
| 11 | `e1e58db7958d` | 5,520 | 2,384 | yes | yes | 0.0000 | AdderBoard Sinusoidal Multi-Depth Mix Experiment | Parameter-free sinusoidal positions provide structured digit offsets while reducing parameters relative to learned absolute positions. |  |
| 12 | `22d1a7729fa3` | 5,520 | 2,384 | yes | yes | 0.0000 | Sinusoidal AdderBoard Depth-Mix Decoder | An equal parameter-free mix of shallow and deep decoder states can preserve direct copying features while retaining deeper carry-processing features. |  |

### OpenEvolve · h12 · RD3

- Run: `oe-size-h12-20260822-b000-rd3-s1`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `80d512944a37` | 2,944 | 2,944 | yes | yes | 0.0000 | Dense Routed Attention-Only Adder | Dense routing of positional and multi-depth attention states can preserve digit identity and carry context without feed-forward sublayers. |  |
| 2 | `3b241581eafd` | 2,944 | 2,944 | yes | yes | 0.0000 | Attention-Only AdderBoard Decoder | Attention-only residual blocks can preserve eligibility without parameter-heavy pointwise feed-forward sublayers. |  |
| 3 | `4661a87a6515` | 6,080 | 2,944 | yes | yes | 0.0000 | Cross-Depth Mixed AdderBoard Decoder | A fixed shallow-deep output mix preserves digit-alignment features while retaining deep carry computations. |  |
| 4 | `8342dd53bf2a` | 5,520 | 2,944 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide parameter-free digit-offset structure for carry alignment. |  |
| 5 | `0f26bc74-00d` | — | 2,944 | no | no | — | Sinusoidal AdderBoard Decoder | Parameter-free sinusoidal positions may provide stable digit-column alignment while reducing trainable parameter count. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 6 | `0c0fda31b07f` | 2,688 | 2,688 | yes | yes | 0.0000 | Attention-Gated Carry Decoder | Replacing the second attention transform with a compact SiLU-gated feed-forward block can model conditional carry interactions with fewer parameters. |  |
| 7 | `4f0aacc01826` | 6,080 | 2,688 | yes | yes | 0.0000 | Gated Carry AdderBoard Decoder | Multiplicative SiLU-gated feed-forward blocks can represent digit-conditioned carry updates more effectively than additive GELU blocks at comparable projection size. |  |
| 8 | `d0373c3feb24` | 5,520 | 2,688 | yes | yes | 0.0000 | Sinusoidal AdderBoard Highway Decoder | A fixed positional-input highway preserves lexical and digit-offset information while the deep branch computes carry context. |  |
| 9 | `6fc06482af06` | 5,056 | 2,688 | yes | yes | 0.0000 | Dual-Attention AdderBoard Decoder | Replacing tokenwise feed-forward sublayers with repeated causal attention may improve carry-context integration while reducing parameters. |  |
| 10 | `b9ee85f83ffc` | 1,888 | 1,888 | yes | yes | 0.0000 | Single-Attention Direct Decoder | A single causal-attention residual block may already expose sufficient digit context, making the tokenwise carry block unnecessary. |  |
| 11 | `b12be3e8c679` | 5,520 | 1,888 | yes | yes | 0.0000 | Sinusoidal Offset AdderBoard Decoder | Fixed sinusoidal positions provide a stable offset basis for digit alignment and carry propagation while eliminating learned positional parameters. |  |
| 12 | `22d1a7729fa3` | 5,520 | 1,888 | yes | yes | 0.0000 | Sinusoidal AdderBoard Depth-Mix Decoder | Mixing the first- and second-block outputs preserves processed digit features without contaminating carry features with an unprocessed embedding bypass. |  |

### AutoResearch · h20 · RD0

- Run: `ar-size-h20-20260822-b000-rd0-s1`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Scalar Pure Context Attention Addition Model | Removing the direct positioned-token shortcut will force tied readout logits to depend entirely on causal attention, preventing local token identity from overwhelming contextual digit and carry information without adding parameters. |  |
| 1 | `30f83d1ee4b5` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Decoder | Fixed sinusoidal positions will provide stable absolute and relative digit-alignment signals while removing learned positional parameters. |  |
| 2 | `bb9fa761be4f` | 5,360 | 5,360 | yes | yes | 0.0000 | Parameter-Free RMS Carry Decoder | Parameter-free RMS normalization will preserve channel-mean arithmetic signals that layer normalization removes while eliminating affine normalization parameters without disrupting causal carry processing. |  |
| 3 | `946e7c3547cb` | 3,824 | 3,824 | yes | yes | 0.0000 | Attention-Only Carry Stage | The first attention and feed-forward block can form aligned digit-sum features, while a parameter-free second attention-only block can route carry information without a redundant second feed-forward transformation. |  |
| 4 | `5a0624c80b8f` | 2,288 | 2,288 | yes | yes | 0.0000 | Two-Stage Attention-Only Addition Model | Two stacked causal attention stages can align operand digits and propagate carry-relevant context using attention softmax nonlinearities, making the remaining feed-forward sublayer unnecessary. |  |
| 5 | `3c212fb51583` | 1,264 | 1,264 | yes | yes | 0.0000 | Single-Stage Attention Addition Model | Because reversed autoregressive output exposes prior result digits as carry context, one causal attention stage may be sufficient to align the current digit with both operands, making the second attention stage redundant. |  |
| 6 | `ddf01ceb4c23` | 568 | 568 | yes | yes | 0.0000 | Compact Attention Gated Digit Model | A compact gated feed-forward stage can nonlinearly map attended operand digits and autoregressive carry context into result-digit features, addressing a capability absent from the zero-accuracy attention-only parent while still reducing parameters. |  |
| 7 | `ccb900a6a55b` | 504 | 504 | yes | yes | 0.0000 | Compact Attention GELU Digit Model | Replacing the gated feed-forward stage with a two-projection GELU stage should retain nonlinear digit computation while removing an unsupported extra gating projection and reducing parameters. |  |
| 8 | `2b98ab41fea5` | 376 | 376 | yes | yes | 0.0000 | Compact Attention-Only Addition Model | Removing the feed-forward residual sublayer should preserve the current public floor because causal attention already mixes aligned digit and carry context, while eliminating parameters that have shown no public accuracy benefit. |  |
| 9 | `71764bce5555` | 46 | 46 | yes | yes | 0.0000 | Two-Channel Attention Bottleneck Addition Model | A two-channel residual stream can preserve the zero public eligibility floor by compressing digit and carry information into a scalar-like state plus an auxiliary channel, while sharply reducing embedding and attention parameters. |  |
| 10 | `f2b3f8747909` | 19 | 46 | no | yes | 0.0000 | Scalar Attention Bottleneck Addition Model | Reducing the residual stream to one scalar channel will preserve the zero public eligibility floor while testing whether a single learned token coordinate plus causal attention is sufficient for the incumbent's observable behavior at substantially lower parameter count. | runtime_transformer_validity |
| 11 | `a293bb3af071` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 12 | `ff47300527b2` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 13 | `5a7ca0612491` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 14 | `2509a5bdff4f` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 15 | `fa399a7a40e3` | 19 | 19 | yes | yes | 0.0000 | Scalar Residual Attention Addition Model | An unnormalized scalar residual stream will preserve direct token identity while causal attention supplies a learned prefix statistic, maintaining the zero public eligibility floor with fewer parameters than the two-channel parent. |  |
| 16 | `70d52c7299fd` | 19 | 19 | yes | yes | 0.0000 | Scalar Sinusoidal Residual Attention Addition Model | A parameter-free sinusoidal position signal will distinguish aligned operand and output locations while the scalar residual preserves token identity, potentially improving addition accuracy without increasing the trusted parameter count. |  |
| 17 | `37ef8249bea6` | 19 | 19 | yes | yes | 0.0000 | Scalar Weighted Context Attention Addition Model | Downweighting the direct positioned-token path while retaining the full causal-attention output will make contextual digit and carry information more influential in tied readout logits without adding parameters. |  |
| 18 | `1ac7bd57eddf` | 19 | 19 | yes | yes | 0.0000 | Scalar Pure Context Attention Addition Model | Removing the direct positioned-token shortcut will force tied readout logits to depend entirely on causal attention, preventing local token identity from overwhelming contextual digit and carry information without adding parameters. |  |
| 19 | `401c88a3c36b` | 46 | 19 | yes | yes | 0.0000 | Two-Dimensional Pure Context Attention Addition Model | Expanding the residual stream from one to two dimensions will provide sinusoidal phase pairs for position-sensitive attention and let the tied readout separate more than two vocabulary classes by hidden-state direction. |  |
| 20 | `faa4194241e8` | 34 | 19 | yes | yes | 0.0000 | Scalar Context Attention with Biased Readout | Adding a bias to the tied scalar readout gives each vocabulary token an independent logit intercept, allowing more than the extreme embedding values to become autoregressive predictions while preserving the contextual attention pathway. |  |

### AutoResearch · h20 · RD1

- Run: `ar-size-h20-20260822-b000-rd1-s1`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Single-Channel Attention-Dominant Mix | An attention-dominant fixed mixture can preserve local token-position identity while retaining strong causal context, improving addition accuracy over a pure scalar-attention bottleneck without adding parameters. |  |
| 1 | `6f0a178fabd6` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Decoder | Parameter-free sinusoidal positions will expose regular digit-offset geometry for operand alignment while removing learned positional parameters. |  |
| 2 | `4be008f72c89` | 5,424 | 5,424 | yes | yes | 0.0000 | Gated Carry Decoder | Replacing each 48-unit GELU feed-forward with a 31-unit SiLU-gated feed-forward will create multiplicative digit-pair and carry conjunctions while slightly reducing parameters. |  |
| 3 | `af03ea700c76` | 3,904 | 3,904 | yes | yes | 0.0000 | Single Gated Carry Decoder | A single gated feed-forward after the first attention can form digit and carry features, while the second attention can route those features directly to readout without a redundant second gated transformation. |  |
| 4 | `2c432b1bc57a` | 2,384 | 2,384 | yes | yes | 0.0000 | Attention Only Carry Decoder | Two causal attention stages can align operand digits and route autoregressive carry evidence without a separate gated feed-forward transformation. |  |
| 5 | `7f84bfa46862` | 1,328 | 1,328 | yes | yes | 0.0000 | Single Attention Carry Decoder | One causal attention stage can directly align operand digits with prior generated digits, preserving autoregressive carry routing without the second attention stage. |  |
| 6 | `2db29de9ef98` | 1,296 | 1,296 | yes | yes | 0.0000 | Single Attention Direct Readout | Removing final normalization lets the tied readout use the shallow residual stream directly while eliminating normalization parameters without disrupting causal digit alignment. |  |
| 7 | `feb8590558d8` | 1,264 | 1,264 | yes | yes | 0.0000 | Single Attention Unnormalized Direct Readout | A single shallow causal attention stage can operate directly on bounded token-plus-sinusoidal representations, making pre-normalization unnecessary and removing its parameters without reducing eligibility. |  |
| 8 | `ad1961f0d780` | 1,044 | 1,044 | yes | yes | 0.0000 | Narrow Attention with Gated Arithmetic | Reallocating parameters from attention width into a small gated feed-forward stage will let attention specialize in digit alignment while multiplicative channel interactions implement local digit and carry transformations, preserving eligibility with fewer parameters. |  |
| 9 | `1d6565e205c9` | 948 | 948 | yes | yes | 0.0000 | Narrow Attention with GELU Arithmetic | Replacing multiplicative gating with a same-size GELU feed-forward stage will preserve the nonlinear digit transformation needed after alignment while removing an unnecessary projection and reducing parameters. |  |
| 10 | `a9b658d05e6d` | 756 | 756 | yes | yes | 0.0000 | Attention-Only Alignment | Removing the pointwise feed-forward stage will preserve the incumbent eligibility if causal attention and the tied embedding readout already provide all publicly demonstrated behavior, while substantially reducing parameters. |  |
| 11 | `0fa8942107cd` | 550 | 550 | yes | yes | 0.0000 | Compressed Attention-Only Alignment | Reducing the residual width from twelve to ten channels will preserve eligibility because causal alignment and token identity require fewer features than the parent allocates, while lowering attention and embedding parameters. |  |
| 12 | `eeb6ae0f6f0b` | 376 | 376 | yes | yes | 0.0000 | Eight-Channel Attention-Only Alignment | Reducing the residual stream from ten to eight channels will preserve eligibility if token identity and causal digit alignment do not require all ten current features, while reducing embedding and attention parameters. |  |
| 13 | `829936167b12` | 234 | 234 | yes | yes | 0.0000 | Six-Channel Attention-Only Alignment | A six-channel residual stream will preserve eligibility because reversed generation externalizes carry state and compact token-position features suffice for causal alignment. |  |
| 14 | `163044325ab0` | 124 | 124 | yes | yes | 0.0000 | Four-Channel Attention-Only Alignment | Four residual channels will preserve parent eligibility because reversed autoregressive addition needs only compact token, alignment, and carry features. |  |
| 15 | `91725b0aeb63` | 46 | 46 | yes | yes | 0.0000 | Two-Channel Dual-Head Attention | Two scalar causal heads can separately track aligned operand evidence and autoregressive carry context, preserving eligibility with fewer embedding and attention parameters. |  |
| 16 | `375db1bb3af7` | 19 | 19 | yes | yes | 0.0000 | Single-Channel Causal Attention | A single scalar causal attention stream can multiplex aligned-digit evidence and carry context sufficiently to preserve the current eligibility floor while reducing parameters. |  |
| 17 | `567fa64ee2b5` | 19 | 19 | yes | yes | 0.0000 | Single-Channel Direct Attention | Removing the residual bypass forces each logit representation to depend on causal context selection, which may improve aligned-digit and carry computation at the same minimum parameter count. |  |
| 18 | `c5b38ba3d292` | 19 | 19 | yes | yes | 0.0000 | Single-Channel Attention-Dominant Mix | An attention-dominant fixed mixture can preserve local token-position identity while retaining strong causal context, improving addition accuracy over a pure scalar-attention bottleneck without adding parameters. |  |
| 19 | `576e8f6a23e5` | — | 19 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 20 | `47c1662a1409` | — | 19 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |

### AutoResearch · h20 · RD2

- Run: `ar-size-h20-20260822-b000-rd2-s1`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Two-Head Width-2 Attention Adder | Two scalar attention heads may retain separate alignment and carry-context routing while width two sharply reduces embedding and attention parameters. |  |
| 1 | `4e0d7301b24b` | 6,080 | 6,080 | yes | yes | 0.0000 | Four-Head AdderBoard Decoder | Increasing each causal attention module from two to four heads will permit separate alignment, operand-retrieval, and carry-context channels without increasing parameter count. |  |
| 2 | `fbb01905a534` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Four-Head Adder Decoder | Fixed sinusoidal positions will expose stable digit-column offsets more directly than learned positions while eliminating positional parameters. |  |
| 3 | `7242eb0d2e48` | 2,384 | 2,384 | yes | yes | 0.0000 | Sinusoidal Attention-Only Adder Decoder | Two causal-attention residual blocks without pointwise feed-forward sublayers will retain the parent's eligibility by retrieving aligned digits and prior carry tokens directly while sharply reducing parameters. |  |
| 4 | `150cbcfa7dae` | 1,328 | 1,328 | yes | yes | 0.0000 | Single Attention-Only Adder Decoder | A single causal-attention residual block can preserve the zero-score parent eligibility floor because direct aligned-digit retrieval does not require a second attention transformation, while removing that block reduces parameters. |  |
| 5 | `2716ac269212` | 1,264 | 1,264 | yes | yes | 0.0000 | Non-Affine RMS Single-Attention Adder | Non-affine RMS normalization can retain attention and logit scale control without learned normalization parameters, preserving the public eligibility floor while reducing model size. |  |
| 6 | `71d408ad96c3` | 756 | 756 | yes | yes | 0.0000 | Three-Head Width-12 Attention Adder | Removing one complete four-dimensional attention head and its associated residual channels will preserve the zero-score public eligibility floor if three heads suffice for alignment and carry context, while substantially reducing parameters. |  |
| 7 | `3bb51fe72c69` | 376 | 376 | yes | yes | 0.0000 | Two-Head Width-8 Attention Adder | Two four-dimensional attention heads may retain sufficient alignment and carry context to preserve the current public eligibility floor while removing the third head and four residual channels. |  |
| 8 | `811d9cca1bd2` | 124 | 124 | yes | yes | 0.0000 | Two-Head Width-4 Attention Adder | Two independent two-dimensional attention heads may preserve separate alignment and carry-context channels while reducing the residual width and learned projection cost. |  |
| 9 | `d88bc61ca0c6` | 46 | 46 | yes | yes | 0.0000 | Two-Head Width-2 Attention Adder | Two scalar attention heads may retain separate alignment and carry-context routing while width two sharply reduces embedding and attention parameters. |  |
| 10 | `a140f2a629da` | 19 | 46 | no | yes | 0.0000 | Single-Head Width-1 Attention Adder | A single scalar causal-attention channel may preserve the zero-valued public eligibility floor while eliminating the second embedding channel and its associated attention projections. | runtime_transformer_validity |
| 11 | `2d8951367a14` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 12 | `11c0359aa94e` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 13 | `eca4be564e75` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 14 | `d5ea73c049a7` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 15 | `f9e457100c60` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 16 | `d2f3b2f8d892` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 17 | `e8707b63001b` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 18 | `40f1b4e0e2ab` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 19 | `bdd54c9a5198` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |
| 20 | `a77555061824` | — | 46 | no | yes | 0.0000 | Unparsed proposal | No structured Architecture IR was recovered. | duplicate_architecture |

### AutoResearch · h20 · RD3

- Run: `ar-size-h20-20260822-b000-rd3-s1`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `6fa79bf1179a` | 6,080 | 6,080 | yes | yes | 0.0000 | Single-Head Bottleneck Addition Decoder | A single eight-dimensional attention head can encode the symmetric operand-digit combination without fragmenting retrieval across heads, preserving eligibility with fewer parameters. |  |
| 1 | `d88e32219924` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Alignment Decoder | Fixed sinusoidal positions will expose relative digit offsets directly, improving operand-output alignment while eliminating the learned positional table. |  |
| 2 | `725caa09c05c` | 3,952 | 3,952 | yes | yes | 0.0000 | Retrieval-First Addition Decoder | Stacking two causal attention retrieval steps before one terminal nonlinear block will preserve digit alignment and carry routing while removing a redundant per-layer feed-forward transformation. |  |
| 3 | `b4d72a09ff16` | 3,904 | 3,904 | yes | yes | 0.0000 | Gated Carry Retrieval Addition Decoder | A compact SiLU-gated terminal feed-forward block will multiplicatively condition digit computation on retrieved carry features, improving arithmetic selectivity while using fewer parameters than the GELU block. |  |
| 4 | `dd4d1e59f68d` | 3,904 | 3,904 | yes | yes | 0.0000 | Interleaved Carry Query Addition Decoder | Moving the existing gated computation between the two attention layers will let nonlinear partial-sum and carry features shape the second layer's queries, improving addition accuracy without increasing parameter count. |  |
| 5 | `14dcb097fa7c` | 3,888 | 3,888 | yes | yes | 0.0000 | Learned Alignment Reallocated Addition Decoder | Reallocating gated feed-forward capacity to learned absolute positions will provide exact fixed-offset digit alignment while retaining enough nonlinear capacity for sums and carries at a lower total parameter count. |  |
| 6 | `a81ca47aefa3` | 3,856 | 3,856 | yes | yes | 0.0000 | Parallel Operand Retrieval Addition Decoder | Parallel causal retrieval branches will specialize independently to the two aligned operands and prior output context, avoiding serial overwriting while a post-mix gated block computes the digit and carry with fewer normalization parameters. |  |
| 7 | `b993bcd4293d` | 3,296 | 3,296 | yes | yes | 0.0000 | Sinusoidal Parallel Retrieval Addition Decoder | Replacing learned absolute positions with parameter-free sinusoidal positions will provide stable digit-offset cues to both causal retrieval branches while reducing parameter count. |  |
| 8 | `8b50b3a078aa` | 3,296 | 3,296 | yes | yes | 0.0000 | Sinusoidal Serial Refinement Addition Decoder | Serializing the two causal-attention modules will let the second module refine aligned operand information and carry state produced by the first, avoiding destructive fixed averaging at unchanged parameter count. |  |
| 9 | `073444374378` | 2,240 | 2,240 | yes | yes | 0.0000 | Sinusoidal Single Retrieval Addition Decoder | One causal attention block should suffice to retrieve aligned operands and prior output carry state, allowing removal of the ineffective second refinement block while preserving eligibility with fewer parameters. |  |
| 10 | `2b620c2d6d3b` | 1,936 | 1,936 | yes | yes | 0.0000 | Sinusoidal GELU Retrieval Addition Decoder | A same-size GELU feed-forward block can transform retrieved operand and carry features without multiplicative gating, reducing parameters and potentially improving optimization while preserving eligibility. |  |
| 11 | `f3dff0e54f87` | 1,328 | 1,328 | yes | yes | 0.0000 | Sinusoidal Identity Retrieval Addition Decoder | Replacing the learned GELU digit-computation transform with identity will preserve retrieved digit features and eligibility while removing parameters if causal attention and autoregressive context already provide the needed computation. |  |
| 12 | `61fc925d715e` | 1,296 | 1,296 | yes | yes | 0.0000 | Sinusoidal Direct Retrieval Addition Decoder | Removing the post-attention normalized identity residual will preserve eligibility while reducing parameters because the causal retrieval residual already contains the digit and carry features needed by the tied readout. |  |
| 13 | `96b17ec18c10` | 1,264 | 1,264 | yes | yes | 0.0000 | Unnormalized Sinusoidal Retrieval Addition Decoder | Removing pre-attention layer normalization will preserve eligibility while reducing parameters because fixed sinusoidal positions and embedding features can directly supply stable digit-alignment cues to causal attention. |  |
| 14 | `0b0006c40cf0` | 1,246 | 1,246 | yes | yes | 0.0000 | Narrow Gated Carry Addition Decoder | Reallocating attention width to a narrow gated post-retrieval transform will preserve eligibility with fewer parameters while supplying the nonlinear digit-and-carry computation absent from the retrieval-only parent. |  |
| 15 | `b29b283a1228` | 1,246 | 1,246 | yes | yes | 0.0000 | Normalized Retrieval Gated Addition Decoder | Parameter-free RMS normalization before causal attention will stabilize content-position scale and make aligned operand retrieval learnable without increasing parameter count. |  |
| 16 | `f17adff05833` | 1,218 | 1,218 | yes | yes | 0.0000 | Normalized Retrieval GELU Addition Decoder | Replacing the narrow gated carry block with a wider GELU bottleneck will preserve useful nonlinear digit computation while removing unnecessary multiplicative-gate parameters. |  |
| 17 | `12aec7263b3b` | 1,070 | 1,070 | yes | yes | 0.0000 | Staged Retrieval Carry Addition Decoder | A second causal attention pass will let the model first encode aligned operand evidence and then integrate prior output context for carry propagation, preserving capability at a narrower width. |  |
| 18 | `0b40e12f0ca8` | 670 | 670 | yes | yes | 0.0000 | Single-Pass Joint Retrieval Addition Decoder | One causal attention pass can jointly retrieve aligned operand digits and prior generated carry evidence, making the second serial attention pass redundant. |  |
| 19 | `444fe53dd1ac` | 550 | 550 | yes | yes | 0.0000 | Attention-Only Retrieval Addition Decoder | Causal attention softmax can provide sufficient nonlinear digit and carry interaction for tied readout, allowing the separate feed-forward computation block to be removed. |  |
| 20 | `4e704e13f586` | 376 | 376 | yes | yes | 0.0000 | Single-Head Bottleneck Addition Decoder | A single eight-dimensional attention head can encode the symmetric operand-digit combination without fragmenting retrieval across heads, preserving eligibility with fewer parameters. |  |

### OpenEvolve · h20 · RD0

- Run: `oe-size-h20-20260822-b000-rd0-s1-r3`
- Condition: Sequential memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions reduce parameters and provide a stable relative-distance signal for carry propagation. |  |
| 2 | `766a52f0beda` | 6,000 | 5,520 | yes | yes | 0.0000 | RMS-Normalized AdderBoard Decoder | RMS normalization can preserve arithmetic signal scale while reducing normalization parameters. |  |
| 3 | `f3b119f9-c93` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide stable arithmetic distance cues while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `8fd0e1624191` | 4,514 | 4,514 | yes | yes | 0.0000 | Cross-Depth Routed AdderBoard Decoder | A learned mix of shallow and deep attention states can replace the second feed-forward sublayer while preserving useful carry information with fewer parameters. |  |
| 5 | `9b15cc1db468` | 6,080 | 4,514 | yes | yes | 0.0000 | AdderBoard SiLU-Gated Decoder | SiLU-gated feed-forward blocks can represent digit and carry interactions more effectively than parameter-matched GELU blocks. |  |
| 6 | `6e3727aa3fe8` | 5,440 | 4,514 | yes | yes | 0.0000 | Sinusoidal RMS-Normalized AdderBoard Decoder | Parameter-free sinusoidal positions can preserve ordered carry propagation while reducing learned positional parameters. |  |
| 7 | `9bc7d6f9-d19` | — | 4,514 | no | no | — | Capacity-Matched Gated Adder Decoder | Capacity-matched SiLU gating improves carry computation through multiplicative channel interactions. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 8 | `9d10ff73624c` | 4,512 | 4,512 | yes | yes | 0.0000 | Fixed Cross-Depth Mix AdderBoard Decoder | A fixed equal mix of shallow and deep states prevents learned routing collapse while preserving both attention paths with fewer parameters. |  |
| 9 | `4661a87a6515` | 6,080 | 4,512 | yes | yes | 0.0000 | Cross-Depth Fixed-Mix AdderBoard Decoder | A parameter-free cross-depth mixture can preserve shallow digit features while retaining deeper carry computation. |  |
| 10 | `7a11d16acd9a` | 3,888 | 3,888 | yes | yes | 0.0000 | Delayed Single-Feed-Forward Causal Decoder | A single feed-forward branch routed around the second attention can preserve nonlinear digit features while reducing redundant parameters. |  |
| 11 | `720b1ce090e6` | 5,984 | 3,888 | yes | yes | 0.0000 | Gated Bottleneck AdderBoard Decoder | Narrow SiLU-gated feed-forward blocks can model carry-dependent interactions with slightly fewer parameters than the GELU blocks. |  |
| 12 | `f727723b7f81` | 3,952 | 3,888 | yes | yes | 0.0000 | Sinusoidal Fixed Cross-Depth Mix AdderBoard Decoder | Parameter-free sinusoidal positions provide stable digit-location signals while reducing parameters relative to learned positional embeddings. |  |
| 13 | `8ec8ff15cb11` | 5,520 | 3,888 | yes | yes | 0.0000 | Sinusoidal Gated AdderBoard Decoder | Parameter-matched gated SiLU feed-forward blocks can represent carry-conditioned transformations more effectively than GELU feed-forward blocks. |  |
| 14 | `0d90f43b64f2` | 5,440 | 3,888 | yes | yes | 0.0000 | Sinusoidal RMS Depth-Mix AdderBoard Decoder | A parameter-free equal mix of shallow and deep representations can improve optimization by preserving direct access to first-block carry features. |  |
| 15 | `f85dd46289a0` | 4,992 | 3,888 | yes | yes | 0.0000 | GELU Bottleneck AdderBoard Decoder | Replacing multiplicative gated feed-forward paths with same-width GELU paths may improve optimization while reducing parameters. |  |
| 16 | `19933b6dca67` | 2,384 | 2,384 | yes | yes | 0.0000 | Attention-Only Cross-Depth Mix AdderBoard Decoder | Removing the feed-forward sublayer will preserve attention-mediated carry propagation while reducing parameters and avoiding an unsupported nonlinear expansion. |  |
| 17 | `bb14aa0c4899` | 3,952 | 2,384 | yes | yes | 0.0000 | Attention-Only Carry Refinement Decoder | An attention-only second stage can refine carry dependencies without the parameter cost or local distortion of a second feed-forward sublayer. |  |
| 18 | `665cf0f774ae` | 3,888 | 2,384 | yes | yes | 0.0000 | Delayed Gated-Feed-Forward Causal Decoder | A narrower SiLU-gated delayed branch can model carry-dependent feature interactions more effectively than a wider additive GELU branch at a similar projection budget. |  |
| 19 | `eaa424805d64` | 5,904 | 2,384 | yes | yes | 0.0000 | RMS Gated Bottleneck AdderBoard Decoder | RMS normalization can preserve the gated decoder's carry computation while reducing normalization parameters by omitting learned centering offsets. |  |
| 20 | `6e3ff277-516` | — | 2,384 | no | no | — | Attention-Only Fixed Cross-Depth Mix AdderBoard Decoder | Removing the unsupported feed-forward sublayer will preserve causal routing capacity while reducing parameter count. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |

### OpenEvolve · h20 · RD1

- Run: `oe-size-h20-20260822-b000-rd1-s1-r3`
- Condition: Sequential memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions can reduce positional overfitting and parameter count while preserving arithmetic position information. |  |
| 2 | `9b15cc1db468` | 6,080 | 5,520 | yes | yes | 0.0000 | Gated Carry Decoder | Parameter-comparable SiLU-gated feed-forward blocks provide multiplicative features that make digit and carry interactions easier to learn. |  |
| 3 | `4543eea2-064` | — | 5,520 | no | no | — | Gated Carry Decoder | Gated SiLU feed-forwards can expose multiplicative digit-and-carry interactions at a projection budget comparable to the GELU control. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `d8ec010b58de` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Multi-Depth AdderBoard Decoder | A parameter-free shallow-deep mixture preserves digit-local features while retaining deep carry computation. |  |
| 5 | `8ec8ff15cb11` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Gated Carry Decoder | Parameter-matched SiLU-gated feed-forward blocks can represent digit and carry interactions more effectively than additive GELU blocks. |  |
| 6 | `de2ea08c7e5f` | 6,080 | 5,520 | yes | yes | 0.0000 | Scaled Residual Carry Decoder | Attenuating every transformed residual branch by one half preserves the token-and-carry stream while still allowing attention and gated updates to influence logits. |  |
| 7 | `8c0af96f-0b5` | — | 5,520 | no | no | — | Sinusoidal Gated Adder Decoder | Matched-budget SiLU-gated feed-forward blocks can model digit and carry interactions more effectively than ordinary GELU blocks. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 8 | `cb88d106a19f` | 6,082 | 5,520 | yes | yes | 0.0000 | Adaptive Multi-Depth AdderBoard Decoder | Learned softmax depth routing can retain shallow digit features without rigidly diluting the deeper carry representation. |  |
| 9 | `bb214006-cb6` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Deterministic sinusoidal positions may expose arithmetic column alignment more reliably while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 10 | `4ccb2ca9-d6a` | — | 5,520 | no | no | — | Sinusoidal Gated Carry Decoder | Fixed sinusoidal positions expose stable relative digit offsets without spending training or parameters on learned absolute position vectors. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 11 | `8dfc2e89-98f` | — | 5,520 | no | no | — | Gated AdderBoard Decoder | SiLU-gated feed-forward blocks can multiplicatively combine attended digit and carry features more effectively than GELU feed-forward blocks. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 12 | `d0bfa8c4bc6c` | 6,592 | 5,520 | yes | yes | 0.0000 | Projected Multi-Depth AdderBoard Decoder | Channel-wise projection of concatenated shallow and deep states can preserve digit-local and carry features that scalar depth mixing collapses. |  |
| 13 | `babc7fa2200e` | 2,384 | 2,384 | yes | yes | 0.0000 | Sinusoidal Attention-Only AdderBoard Decoder | Removing pointwise feed-forward sublayers may preserve carry routing while reducing optimization interference and parameter count. |  |
| 14 | `f5bc75e79de5` | 6,080 | 2,384 | yes | yes | 0.0000 | Selective Residual Carry Decoder | Restoring unit-strength attention residuals while retaining half-strength feed-forward updates preserves operand retrieval without allowing local nonlinear updates to overwhelm the carry stream. |  |
| 15 | `f5d3043765cf` | 3,952 | 2,384 | yes | yes | 0.0000 | Fixed-Mix Attention Refinement Decoder | A parameter-free cross-depth mix can preserve local digit information while a lightweight second attention refinement propagates carries, making the second gated feed-forward block unnecessary. |  |
| 16 | `a7e03f9f-996` | — | 2,384 | no | no | — | Channel-Selective Multi-Depth AdderBoard Decoder | Concatenating shallow and deep states before projection can preserve different digit and carry channels that scalar depth routing must mix uniformly. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 17 | `72cfc9289818` | 5,520 | 2,384 | yes | yes | 0.0000 | Sinusoidal Anchored AdderBoard Decoder | A parameter-free late anchor to positioned token embeddings can preserve digit identity while contextual attention computes carries. |  |
| 18 | `96a60a123384` | 5,520 | 2,384 | yes | yes | 0.0000 | Sinusoidal Scaled Residual Carry Decoder | A parameter-free sinusoidal coordinate system supplies ordered position structure more efficiently than independently learned position vectors. |  |
| 19 | `31d2a142de9e` | 3,952 | 2,384 | yes | yes | 0.0000 | Direct Attention Refinement Decoder | Directly reading the second attention-residual state avoids diluting carry corrections with a redundant shallow-state mixture. |  |
| 20 | `9cb09fe3-ee0` | — | 2,384 | no | no | — | Concat-Projected Multi-Depth AdderBoard Decoder | Concatenating shallow and deep states before projection permits channel-selective retention of digit and carry features that scalar depth mixing cannot express. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |

### OpenEvolve · h20 · RD2

- Run: `oe-size-h20-20260822-b000-rd2-s1-r2`
- Condition: Portfolio memory · neutral review

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Parameter-free sinusoidal positions preserve eligibility while providing structured distance signals for arithmetic. |  |
| 2 | `d8ec010b58de` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Depth Mix AdderBoard Decoder | Equal fixed mixing of shallow and deep states preserves alignment features while retaining carry refinement. |  |
| 3 | `34b4f2b8-a77` | — | 5,520 | no | no | — | Sinusoidal AdderBoard Decoder | Fixed sinusoidal positions provide stable digit-distance structure while removing learned positional parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 4 | `9b15cc1db468` | 6,080 | 5,520 | yes | yes | 0.0000 | Gated Feed-Forward AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks improve learned carry transformations. |  |
| 5 | `30feb319ca36` | 6,080 | 5,520 | yes | yes | 0.0000 | Fixed Input-Depth Mix AdderBoard Decoder | A parameter-free input-depth mix preserves token and position identity while retaining contextual carry computation. |  |
| 6 | `22d1a7729fa3` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Fixed Depth Mix AdderBoard Decoder | Fixed sinusoidal positions reduce parameters and provide stable relative phase cues for digit alignment. |  |
| 7 | `042df8b4c2a3` | 5,920 | 5,520 | yes | yes | 0.0000 | RMS-Normalized AdderBoard Decoder | Parameter-free RMS normalization may preserve eligibility while improving scale stability and reducing parameter count. |  |
| 8 | `8ec8ff15cb11` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal Gated AdderBoard Decoder | Fixed sinusoidal positions reduce parameters and provide a regular positional basis for digit alignment and carry propagation. |  |
| 9 | `4c1878174542` | 6,082 | 5,520 | yes | yes | 0.0000 | Adaptive Input-Depth Mix AdderBoard Decoder | A learned softmax depth mix can preserve digit identity while adapting how strongly contextual carry features influence the readout. |  |
| 10 | `4deab172-7c5` | — | 5,520 | no | no | — | Sinusoidal Full Depth AdderBoard Decoder | Reading the second block directly avoids attenuating its learned carry-processing update through equal depth averaging. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 11 | `57e8e79c6032` | 5,920 | 5,520 | yes | yes | 0.0000 | Cross-Depth Fixed-Mix AdderBoard Decoder | A parameter-free fixed mix of first- and second-block representations may preserve local digit features while retaining deeper carry computations. |  |
| 12 | `82b119df4425` | 5,520 | 5,520 | yes | yes | 0.0000 | Fixed Layer-Mix Sinusoidal Gated Adder Decoder | Parameter-free averaging of shallow and deep causal states can preserve direct digit features while retaining two-layer carry computation. |  |
| 13 | `3a5abc7ea247` | 6,082 | 5,520 | yes | yes | 0.0000 | SiLU-Gated Carry AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks can represent digit-and-carry interactions more effectively than additive GELU transforms. |  |
| 14 | `31d21e60-39f` | — | 5,520 | no | no | — | Sinusoidal Deep-State AdderBoard Decoder | Reading only the deepest residual state avoids diluting second-block carry refinements with an equal mixture of the earlier state. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 15 | `be0a22523183` | 5,920 | 5,520 | yes | yes | 0.0000 | SiLU-Gated AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks may model digit and carry interactions more effectively than GELU feed-forward blocks. |  |
| 16 | `b95f680f3655` | 4,496 | 4,496 | yes | yes | 0.0000 | Fixed Layer-Mix Sinusoidal GELU Adder Decoder | Ungated GELU feed-forward blocks may preserve useful nonlinear processing while reducing parameters relative to multiplicative gated blocks. |  |
| 17 | `2ae49904-0ee` | — | 4,496 | no | no | — | Sinusoidal Depth-Mix AdderBoard Decoder | An equal fixed mixture of shallow and deep representations preserves digit-local features while retaining contextual carry computation. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 18 | `80b11dfa-c27` | — | 4,496 | no | no | — | Sinusoidal Gated Feed-Forward AdderBoard Decoder | Parameter-matched SiLU-gated feed-forward blocks can represent conditional carry transformations more effectively than additive GELU blocks. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 19 | `4b2dc7f15330` | 2,848 | 2,848 | yes | yes | 0.0000 | Attention-Only RMS Adder Decoder | Removing both feed-forward sublayers may preserve causal token mixing while shortening optimization paths and substantially reducing parameters. |  |
| 20 | `babc7fa2200e` | 2,384 | 2,384 | yes | yes | 0.0000 | Sinusoidal Attention-Only AdderBoard Decoder | Removing feed-forward sublayers preserves causal token mixing while eliminating nonlinear parameters that have shown no public accuracy benefit. |  |

### OpenEvolve · h20 · RD3

- Run: `oe-size-h20-20260822-b000-rd3-s1`
- Condition: Portfolio memory · assumption challenge

| Iteration | Candidate | Parameters | Best valid | Valid | New architecture | Public accuracy | Architecture | Mechanism / next experiment | Failure |
|---:|---|---:|---:|:---:|:---:|---:|---|---|---|
| 0 | `7a8edd5d72d1` | 6,080 | 6,080 | yes | yes | 0.0000 | Conventional AdderBoard Starting Architecture | Conventional causal decoder control architecture |  |
| 1 | `d8ec010b58de` | 6,080 | 6,080 | yes | yes | 0.0000 | Fixed Cross-Depth Mix AdderBoard Decoder | A parameter-free average of first- and second-block representations preserves useful carry features that serial depth may overwrite. |  |
| 2 | `8342dd53bf2a` | 5,520 | 5,520 | yes | yes | 0.0000 | Sinusoidal AdderBoard Decoder | Deterministic sinusoidal positions can expose digit distance structure while removing learned positional parameters. |  |
| 3 | `1fb58d494f10` | 6,080 | 5,520 | yes | yes | 0.0000 | AdderBoard Decoder with Shallow Context Bypass | A parameter-free fixed bypass of token-position features preserves digit identity while the deep stream computes carries. |  |
| 4 | `3b241581eafd` | 2,944 | 2,944 | yes | yes | 0.0000 | Attention-Only AdderBoard Decoder | Removing tokenwise feed-forward branches preserves causal carry routing while reducing capacity and optimization friction. |  |
| 5 | `9b15cc1db468` | 6,080 | 2,944 | yes | yes | 0.0000 | Gated Carry AdderBoard Decoder | SiLU-gated feed-forward blocks can represent conditional carry operations more effectively than parameter-matched GELU blocks. |  |
| 6 | `a4887b30e056` | 5,520 | 2,944 | yes | yes | 0.0000 | Gated Carry Decoder | Parameter-matched multiplicative GELU-gated feed-forwards can model digit-context carry interactions more efficiently than standard GELU MLPs. |  |
| 7 | `26daae8df725` | 5,520 | 2,944 | yes | yes | 0.0000 | AdderBoard Decoder with Sinusoidal Shallow Bypass | Deterministic sinusoidal positions provide an immediately usable arithmetic alignment signal while removing learned positional parameters. |  |
| 8 | `a4c0c384b86a` | 2,304 | 2,304 | yes | yes | 0.0000 | Gated Carry AdderBoard Decoder | A compact gated tokenwise stage can transform routed digit context into carry state more effectively and with fewer parameters than a second attention stage. |  |
| 9 | `22d1a7729fa3` | 5,520 | 2,304 | yes | yes | 0.0000 | Sinusoidal Cross-Depth Mix AdderBoard Decoder | Parameter-free sinusoidal positions can provide sufficient digit-order structure while reducing parameters and avoiding unsupported learned positional adaptation. |  |
| 10 | `babc7fa2200e` | 2,384 | 2,304 | yes | yes | 0.0000 | Attention-Only Carry Decoder | Removing pointwise feed-forwards will preserve causal carry routing while reducing parameters and possible optimization interference. |  |
| 11 | `9d728e40-bb9` | — | 2,304 | no | no | — | AdderBoard Decoder with Context-Only Readout | Removing the fixed shallow-feature bypass prevents raw positional and token features from diluting the carry-sensitive contextual representation at readout. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 12 | `5a7e1b329102` | 1,744 | 1,744 | yes | yes | 0.0000 | Sinusoidal Gated Carry Decoder | Fixed sinusoidal positions can preserve sequence-order information for addition while eliminating learned positional parameters. |  |
| 13 | `8ec8ff15cb11` | 5,520 | 1,744 | yes | yes | 0.0000 | Sinusoidal Gated Carry AdderBoard Decoder | Fixed sinusoidal positions can preserve digit-offset alignment while eliminating unnecessary learned position parameters. |  |
| 14 | `4e8ab5b2829f` | 1,328 | 1,328 | yes | yes | 0.0000 | Single-Attention Carry Decoder | One full-prefix causal attention stage may route carry information directly, making the second attention stage redundant. |  |
| 15 | `4e6051a4-86f` | — | 1,328 | no | no | — | AdderBoard Decoder with Deep-Only Readout | Removing the fixed shallow bypass prevents unprocessed positional features from diluting carry-sensitive deep representations at readout. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 16 | `4e1b880be2ac` | 1,616 | 1,328 | yes | yes | 0.0000 | Sinusoidal GELU Carry Decoder | A plain GELU residual carry transform can preserve useful nonlinear features with fewer parameters and less multiplicative suppression than a gated transform. |  |
| 17 | `82b119df4425` | 5,520 | 1,328 | yes | yes | 0.0000 | Sinusoidal Fixed Depth-Mix AdderBoard Decoder | A parameter-free equal mix of first- and second-block states can preserve early digit features while retaining deeper carry computation. |  |
| 18 | `a5d8861a5d11` | 2,288 | 1,328 | yes | yes | 0.0000 | Non-Affine RMS Attention-Only Carry Decoder | Non-affine RMS normalization can stabilize causal attention without the centering and learned affine parameters of layer normalization. |  |
| 19 | `185715e8-9e8` | — | 1,328 | no | no | — | AdderBoard Decoder with Sinusoidal Deep Readout | Removing the fixed shallow-feature bypass prevents raw positional features from diluting the contextual carry representation presented to the readout. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |
| 20 | `a08685cd-e39` | — | 1,328 | no | no | — | Sinusoidal Attention-Only Decoder | Causal attention already supplies the contextual carry signal, so removing the pointwise carry block can preserve eligibility with fewer parameters. | OpenEvolve proposal duplicates an architecture already proposed or evaluated in this run |

## Reading the dashboard

The dashboard plots iteration on the x-axis and the running best structurally valid parameter count on the y-axis. Toggle horizons, frameworks, and intervention conditions to compare search dynamics rather than only terminal winners. The proposal table exposes the underlying architectures and failure/duplication events.
