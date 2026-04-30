# Run the CGAL/Monty benchmark on Azure (single GPU spot VM)

This subdirectory contains a Bicep template + helper scripts that
provision a GPU spot VM, install Docker + NVIDIA toolkit, clone this
repo + the Monty fork, download YCB data, and run the comparison
matrix automatically. Results are pulled back via `scp`.

## Why Azure?

- `habitat-sim` ships **no arm64 conda builds** — running the Docker
  stack on an Apple Silicon Mac means QEMU emulation (~5–10× slower).
- A T4 GPU makes habitat rendering 10–50× faster than CPU. A full
  matrix that takes hours on CPU finishes in well under an hour.
- Spot pricing on `Standard_NC4as_T4_v3` is roughly $0.20–0.40/h.

## Files

| File | Purpose |
| --- | --- |
| [main.bicep](main.bicep) | VM + vnet + NSG + cloud-init |
| [deploy.sh](deploy.sh) | One-shot: quota check → deployment → watch/fetch hints |
| [watch.sh](watch.sh) | Tail `/var/log/cgal-bench.log` on the VM |
| [fetch-results.sh](fetch-results.sh) | `scp` `docker/logs/` back into `./azure-results` |

## Prerequisites

```bash
brew install azure-cli jq
az login
az account set --subscription <id>
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519     # if you don't have one
```

### Quota — check before you deploy

The default SKU (`Standard_NC4as_T4_v3`) is in the
`standardNCASv3T4Family` quota family and is **zero by default in most
new subscriptions**. If your portal denies the request in one region
(e.g. North Europe T4 is heavily contested), survey alternatives:

```bash
./find-quota.sh
# or with custom regions:
REGIONS="swedencentral westeurope eastus2" ./find-quota.sh
```

This prints a `FAMILY × REGION` matrix with `YES` next to anything
that has spare vCPUs ≥ what the SKU needs. Pick the first `YES` and
deploy:

```bash
LOCATION=<region> VM_SIZE=<sku> ./deploy.sh
```

Common workable substitutes when T4 is denied:

| If denied | Try | Reason |
| --- | --- | --- |
| `Standard_NC4as_T4_v3` in NE | same SKU in `swedencentral` / `westeurope` / `uksouth` / `eastus2` | T4 quota is region-specific |
| any T4 region | `Standard_NV6ads_A10_v5` (A10 GPU, family `standardNVADSA10v5Family`) | usually less contended, similar perf |
| any GPU | `Standard_NC6s_v3` (V100, family `standardNCSv3Family`) | older but plentiful |
| no GPU quota at all | `Standard_D8s_v5` (CPU only) | unlimited quota, much slower |

If you'd rather not chase quota at all, run on CPU:

```bash
VM_SIZE=Standard_D8s_v5 ./deploy.sh
```

## Deploy

```bash
RG=cgal-bench LOCATION=swedencentral SUITE=noise ./deploy.sh
```

Defaults:

| Var | Default | Notes |
| --- | --- | --- |
| `RG` | `cgal-bench` | Resource group (created if missing) |
| `LOCATION` | `swedencentral` | Region with cheap NCASv3 T4 quota |
| `VM_SIZE` | `Standard_NC4as_T4_v3` | T4 GPU, 4 vCPU, 28 GB |
| `USE_SPOT` | `true` | Spot eviction = OS disk deleted |
| `SUITE` | `noise` | `smoke` / `noise` / `full` (see [docker/](../docker/README.md)) |
| `SSH_PUB` | `~/.ssh/id_ed25519.pub` | Public key for the `cgal` user |

The VM bootstraps in ~5 min, then runs the benchmark for ~30–90 min
depending on `SUITE`. Watch progress:

```bash
./watch.sh           # tails /var/log/cgal-bench.log
```

## Get results

```bash
./fetch-results.sh                      # → ./azure-results/COMPARISON_REPORT.md
open ./azure-results/COMPARISON_REPORT.md
```

The directory contains:

- `COMPARISON_REPORT.md` — auto-generated baseline-vs-CGAL markdown
  report (summary + deltas + per-experiment expansion).
- `<experiment>/<arm>/eval_stats.csv`, `train_stats.csv`, `run.log` —
  Monty's own logs.

## Tear down

```bash
az group delete -n cgal-bench --yes --no-wait
```

Spot VMs already auto-delete their OS disk on eviction; deleting the
RG removes the public IP, NIC, vnet, NSG and any leftover state.

## Notes / caveats

- The image runs a non-headless habitat-sim build is **not** used —
  we keep the headless build that uses EGL on the GPU. Works fine on
  T4 / V100 with the standard NVIDIA driver.
- Cloud-init logs go to `/var/log/cloud-init-output.log`; the
  benchmark log is `/var/log/cgal-bench.log`.
- If the matrix fails partway through, the report is still emitted
  with `n/a` rows for the missing arms — re-run a single experiment
  with `EXPERIMENTS=...` from inside `/opt/cgal/docker`.
- Spot eviction will lose the run. Re-deploy with `USE_SPOT=false` if
  that's a concern.
