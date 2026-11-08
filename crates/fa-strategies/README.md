# fa-strategies

Strategy population: GP programs and neural policies producing exposure distributions.

**Status:** partial — GP DSL AST + interpreter, `GpStrategy`, `Strategy`/`NeuralPolicy` traits, proto conversion; S-expression program syntax (`sexpr`), `strategies.toml` population files and the hand-written **baseline** set (`baselines.toml`, not evolved), and a price-only fallback observation (`features::FallbackContext`) used by `bins/strategy-runtime` until graph/decoder feeds exist (all tested). Neural policies, evolution operators and random program generation are pending.

## Owns
GP DSL interpreter, policy nets, `Strategy` trait.

## Governing specs
- [35-strategies](../../specs/35-strategies.md)
