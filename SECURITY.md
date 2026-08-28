# Security Policy

## Reporting a vulnerability

Report privately through GitHub's **Report a vulnerability** button under the
Security tab of this repository, which opens a private advisory visible only to
maintainers. Please do not open a public issue for a security problem.

Include what you would put in a bug report — GPU, driver, platform, command —
plus what an attacker could achieve.

## Scope

Pantheon is a hardware stress tool. Things that are expected behaviour rather
than vulnerabilities:

- **High power draw, heat and clock throttling.** Driving hardware to its limits
  is the purpose of these workloads.
- **A workload that hangs or resets the GPU when given extreme parameters.**
  Launch parameters are bounded to make this hard, but a sufficiently large
  value is still a long-running kernel.
- **Reported faults on failing hardware.** A `Verification: FAIL` means the GPU
  produced a wrong result. That is a finding about the hardware, not about
  Pantheon.

In scope: anything letting a crafted input, report file or workload name cause
Pantheon to execute unintended code, escape its own process, write outside its
working directory, or exfiltrate data from the host it runs on.

## What Pantheon collects

Runs record GPU telemetry and RAS/ECC counters locally under `results/` and
`database/`. Nothing is transmitted anywhere. If you publish reports, note that
they contain GPU UUIDs and serial numbers unless you remove them.
