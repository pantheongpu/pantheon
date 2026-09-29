# ReFrame

[`pantheon_check.py`](pantheon_check.py) is a test for
[ReFrame](https://github.com/reframe-hpc/reframe). It runs one Pantheon
workload on the GPUs of a node and reports the score and the highest
temperature of each card as performance values, so a card that got slower
shows up against the reference of its node type.

```bash
reframe -c pantheon_check.py -r
```

The test runs on partitions with the feature `gpu` and in environments with
the feature `cuda` or `hip`. [`settings_example.py`](settings_example.py) is
a configuration for one machine that declares them:

```console
$ reframe -C settings_example.py -c pantheon_check.py -r --exec-policy serial -S devices=0 -S duration=5 --performance-report
P: gpu0_score: 336.62 GB/s
P: gpu0_max_temp: 57.0 C
P: gpu0_score: 2894590000.0 march-ops/s
P: gpu0_max_temp: 55.0 C
P: gpu0_score: 3373.0 retained-MiB
P: gpu0_max_temp: 51.0 C
P: gpu0_score: 4.10915 TFLOPS
P: gpu0_max_temp: 58.0 C
[  PASSED  ] Ran 4/4 test case(s) from 4 check(s) (0 failure(s), 0 expected failure(s), 0 skipped, 0 aborted)
```

| Variable | Default | Meaning |
|---|---|---|
| `workload` (parameter) | `memory_read`, `march_test`, `memory_retention`, `tensor_virus` | One test for each |
| `duration` | 60 | Seconds for each workload |
| `devices` | all | The cards to test, for example `-S devices=0,1` |
| `mem_percent` | 99 | Percentage of the free memory of a card |
| `platform` | `auto` | `mock` runs on the CPU, to try the test without a GPU |

The test fails when a card did not complete the workload, when Pantheon's
verification found an error, when the card counted uncorrectable errors, and
when Pantheon ran on its CPU backend although nobody asked for it.

Use `--exec-policy serial` on a machine without a scheduler. ReFrame
otherwise starts the four workloads on the same card at the same time.
