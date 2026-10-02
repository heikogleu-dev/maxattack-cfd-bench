<!-- Thanks for opening a PR! Please fill in the relevant sections. -->

## Summary

<!-- 1-3 sentences: what this changes and why -->

## Type

- [ ] Solver / kernel change (`src/`)
- [ ] Analysis or run tooling (`werkzeuge/`)
- [ ] New or updated measurement
- [ ] Documentation fix or improvement
- [ ] Other:

## The one variable this changes

<!-- Name it. Everything else must be identical to the reference run. -->

## Evidence

- Action-path counter(s) and their `is = should` lines from the run log:
- Scratch gate (`werkzeuge/scratch_gate/scratch_gate.sh`) result, if kernels changed:
- Runs (name, commit, device, `CFD_*` line):

## Hardware / Software (if a measurement)

- GPU:
- CPU + RAM:
- Kernel / `xe` driver:
- Intel Compute Runtime:

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] My contribution is licensed under [LICENSE.md](../LICENSE.md) (FluidX3D license, non-commercial, unaltered)
- [ ] Numbers come from field data / probes, not from rendered images
- [ ] New kernels went CPU → iGPU → discrete GPU
