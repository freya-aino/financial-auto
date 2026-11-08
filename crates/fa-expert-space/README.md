# fa-expert-space

MoE router, expert spaces, residual-decorrelation objective, decoder-facing contract.

**Status:** partial — `ExpertView` contract, `ExpertSpaceConfig`, the `Router` stage trait with the baseline `EvidenceSoftmaxRouter` (softmax over per-expert energy or confidence), residual-correlation penalty (D-008) and router load-balance term; all tested. The learned MoE gate lands with the `burn` backend behind `Router`.

## Owns
`ExpertView` contract consumed by decoders and strategies; `Router` stage boundary.

## Governing specs
- [32-expert-spaces](../../specs/32-expert-spaces.md)
