# Prometheus

The node check writes its result as metrics for
[node_exporter's textfile collector](https://github.com/prometheus/node_exporter#textfile-collector),
so a verdict shows up in Grafana beside the numbers a site already collects.

```bash
pantheon_node_check.py --test march_test --duration 30 \
    --textfile /var/lib/node_exporter/textfile_collector/pantheon.prom
```

The file is written in one step, so a scrape never sees half of it. What it
holds, from a run on two cards where one counted correctable errors:

```
# HELP pantheon_verdict_code Verdict of the last Pantheon node check: 0 healthy, 1 watch, 2 fault, 3 nothing was tested.
# TYPE pantheon_verdict_code gauge
pantheon_verdict_code 1
pantheon_verdict_info{verdict="WATCH"} 1
pantheon_gpu_verdict_code{gpu="0",name="NVIDIA H100 PCIe"} 0
pantheon_gpu_verdict_code{gpu="1",name="NVIDIA H100 PCIe"} 1
pantheon_gpu_score{gpu="0",workload="memory_read",unit="GB/s"} 1971.4
pantheon_gpu_max_temperature_celsius{gpu="0",workload="memory_read"} 72
pantheon_gpu_max_power_watts{gpu="0",workload="memory_read"} 288.5
pantheon_gpu_workload_failed{gpu="1",workload="march_test"} 0
pantheon_last_run_timestamp_seconds 1790800000
```

| Metric | Meaning |
|---|---|
| `pantheon_verdict_code` | The node: 0 healthy, 1 watch, 2 fault, 3 nothing was tested |
| `pantheon_gpu_verdict_code` | The same for each card |
| `pantheon_gpu_score` | The score of each workload on each card, in the unit of the label |
| `pantheon_gpu_max_temperature_celsius`, `pantheon_gpu_max_power_watts` | The highest reading during each workload |
| `pantheon_gpu_workload_failed` | 1 when the workload failed or did not complete on the card |
| `pantheon_last_run_timestamp_seconds` | When the check last finished, to alert on a check that stopped running |

## Running it on a schedule

A cron entry on each GPU node, for a check on the hour:

```
0 * * * * root /usr/local/sbin/pantheon_node_check.py --test march_test --duration 30 --textfile /var/lib/node_exporter/textfile_collector/pantheon.prom >/dev/null 2>&1
```

The check loads the cards for the duration, so on a node that runs jobs, run
it from the scheduler instead: the [Slurm epilog](../slurm/) can add
`--textfile` to its node check, and the metrics then describe the cards after
each job.

## Alerts

```yaml
groups:
  - name: pantheon
    rules:
      - alert: GPUFault
        expr: pantheon_gpu_verdict_code == 2
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "GPU {{ $labels.gpu }} on {{ $labels.instance }} has a fault"
      - alert: GPUWatch
        expr: pantheon_gpu_verdict_code == 1
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "GPU {{ $labels.gpu }} on {{ $labels.instance }} needs a look"
      - alert: PantheonCheckStale
        expr: time() - pantheon_last_run_timestamp_seconds > 7200
        labels:
          severity: warning
        annotations:
          summary: "No Pantheon check on {{ $labels.instance }} for two hours"
```
