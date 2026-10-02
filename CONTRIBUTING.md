# Contributing

Thanks for considering a contribution. This repository is a **modified version of FluidX3D**
(© Dr. Moritz Lehmann), extended with a two-domain coupling and a facet wall model for vehicle
aerodynamics at 4 mm on a single Intel Arc Pro B70. It is not FluidX3D and not endorsed by its author.

## License first

Every contribution falls under **[LICENSE.md](LICENSE.md)**, the unaltered FluidX3D license. By
contributing you accept its terms, in particular:

- non-commercial use only, no military or defence use;
- no training of AI models on the source code;
- altered versions must be marked as such and their source published;
- the license notice must not be removed or altered.

Do not submit code you cannot license under these terms.

Bugs in upstream FluidX3D itself belong to <https://github.com/ProjectPhysX/FluidX3D/issues>.

## High-value contributions

- **Reproducing a result on other hardware** (other Arc / Battlemage cards, other vendors): open an
  issue with your numbers, the exact `CFD_*` line and the commit hash.
- **A different result on the same hardware**: driver, kernel and Compute Runtime versions are data.
- **GPU hangs, `CL_OUT_OF_RESOURCES`, xe timeouts**: include the run log, the `CFD_*` line and the
  `journalctl -k` lines around the event.
- **Independent references** (OpenFOAM, experiments, literature) for the vehicle or the channel cases.

## How to contribute

1. **Open an issue first** for anything non-trivial — aligning on direction is faster than reworking a PR.
2. Fork, branch, edit, push, open a PR against `master`.
3. **One variable per change.** A PR that changes physics and output at the same time cannot be judged.
4. Every new mechanism needs an **action-path counter** with a declared target, checked as
   `is = should` in the run log. A switch whose counter does not fire is a hard error.
5. Kernel changes: run `werkzeuge/scratch_gate/scratch_gate.sh` (offline `ocloc`, no GPU run) and state
   the result. New kernels go **CPU → iGPU → discrete GPU**, never straight onto the desktop GPU.

## Reporting style

- Real numbers from runs, not impressions; name the run, the commit and the device.
- Measure on field data and probes (CSV/VTK), not on rendered images.
- Honest verdicts — negative results are kept on record and are valuable.
- Do not validate physics against an earlier version of this code; valid references are independent
  implementations (e.g. OpenFOAM) and the literature.
