---
name: Bug report
about: A workload behaves incorrectly, or the runner fails
title: ''
labels: bug
---

**What happened**

**What you expected**

**Command**
```bash
python3 pantheon.py --test ... --duration ... --gpu ...
```

**Output**
```
paste the relevant output, including any Verification: or [SDC FAULT] line
```

**Environment**

| | |
| --- | --- |
| GPU | e.g. RTX 4090, MI300X |
| Driver | `nvidia-smi` / `rocm-smi` version |
| Platform | CUDA / HIP / MOCK |
| Toolkit | CUDA or ROCm version |
| OS | |
| Pantheon | `python3 pantheon.py --version` |

**Does it reproduce under `PLATFORM=MOCK`?**

This matters: the mock backend needs no GPU, so a bug that reproduces there is
far quicker to fix. A bug that does *not* reproduce there is likely
hardware-, driver- or concurrency-specific — please say so.
