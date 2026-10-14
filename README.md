# nebius-gpu-lab

A GPU-on-Kubernetes lab environment I built on Nebius Managed Kubernetes to
work through the operational layers between a PCIe device and a schedulable
`nvidia.com/gpu`, and the ways each of them fails.

Not production configuration. These are deliberately small, single-node
manifests meant to be broken on purpose and watched.

| File | What it is |
|---|---|
| `gpu-operator-values.yaml` | NVIDIA GPU Operator values, driver and toolkit enabled so the operator builds the driver on a bare node |
| `kps-values.yaml` | kube-prometheus-stack values, with the scrape config that picks up the DCGM exporter |
| `dcgm-metrics.csv` | Custom DCGM field list, including the profiling fields the exporter leaves out by default |
| `alerts.yaml` | Three PrometheusRules: Xid, throttling under load, and exporter down |
| `load.yaml` | A GPU workload big enough to move every panel on a dashboard |
| `training/` | Single-GPU and two-rank DDP jobs, with knobs for batch size, dataloader workers and checkpoint interval, for reproducing OOM, input-pipeline starvation and mid-run node loss |
| `inference/` | vLLM deployment and a concurrency load generator, for finding the saturation knee and KV cache starvation |

Set `adminPassword` in `kps-values.yaml` before installing anything.

## What it is for

Each piece exists to make one failure reproducible on demand:

- A GPU node with no driver, so you can watch the operator build the stack and
  see the order the pods have to come up in.
- Time-slicing versus MIG, so the difference in memory isolation is something
  you have observed rather than read.
- A job that can be killed mid-run, so checkpoint interval and restart cost are
  numbers rather than theory.
- An inference server that can be starved of KV cache while GPU utilization
  barely moves, which is the shape of most "make it faster" tickets.
