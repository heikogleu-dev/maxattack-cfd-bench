# Security Policy

## Scope

This repository contains a GPU flow solver (C++ host code and OpenCL kernels), analysis scripts and
documentation. It runs no network services.

## What counts as a security issue

- Leaked credentials, tokens or personal data in a committed file
- A documented setup step or script that creates a security risk on the user's machine if followed
- Code that reads or writes outside the documented run, log and export directories
- Malicious links in documentation

A GPU hang, a driver reset or a `CL_OUT_OF_RESOURCES` abort is a **bug**, not a security issue —
please report it as a regular issue with the run log and the `journalctl -k` lines.

## Reporting

Please **do not** open a public issue for security findings.

Contact the maintainer via the email address listed on the GitHub profile
(<https://github.com/heikogleu-dev>), or use GitHub's private advisory mechanism on this repository:

→ Security tab → Report a vulnerability

You will receive an acknowledgement within 7 days.

## Out of scope

Issues in upstream projects (FluidX3D, Intel Compute Runtime, the Linux `xe` driver, OpenCL headers)
should be reported to those projects directly. If you are unsure where a problem belongs, open a
regular issue and we will help triage.
