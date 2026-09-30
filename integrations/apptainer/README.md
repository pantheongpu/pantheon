# Apptainer

[`pantheon.def`](pantheon.def) builds a SIF image from the published container
image, for sites that run containers through Apptainer or Singularity.

```bash
apptainer build pantheon.sif pantheon.def
apptainer run --nv pantheon.sif --test march_test --duration 60 --gpu all
```

The image carries the CUDA 12.8 toolchain, so the node needs only its driver.
`--nv` hands the driver to the container. Pantheon compiles its workloads on
the first run into `$HOME/.cache/pantheongpu/builds`, which Apptainer binds
from the host, so the compiled workloads survive between runs. Reports land in
`./database` of the directory you run from.

For AMD nodes, change the `From:` line to the `1.2.2-rocm6.4` tag and run
with `--rocm`.

The image is about 5 GB, because it carries the compiler.
