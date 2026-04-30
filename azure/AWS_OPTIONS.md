# AWS as an alternative to Azure for the CGAL benchmark

Written after the Azure DSv5 quota approval (4 vCPU only) blocked the full
matrix. Question: does moving to AWS make the situation materially better?

## TL;DR

**No — not for a fresh AWS account.** AWS will impose the same kind of quota
song-and-dance Azure just did. New accounts ship with **0 vCPU** of quota for
every GPU family ("Running On-Demand G and VT instances", "Running On-Demand
P instances", etc.) and a modest CPU quota (~32 vCPU on-demand standard).
You will file Service Quota tickets and wait, exactly like with Azure.

If you want a GPU **today** without quota theatre, skip the hyperscalers:

- **Lambda Labs** — `gpu_1x_a10` is ~$0.75/h, click-to-launch, no quota
  approval, Ubuntu 22 with NVIDIA drivers preinstalled. Best fit.
- **RunPod / Vast.ai** — spot-style A10/T4/L4 from ~$0.20/h, but you bring
  your own Docker image and storage hygiene is on you.
- **Paperspace Gradient** — A4000/A5000 from ~$0.45/h, persistent storage
  built-in, similar UX to Lambda.

If you want to stay on a hyperscaler and have an existing AWS account with
some history, AWS is *roughly* equivalent to Azure on price; below is the
detail and the migration cost.

## AWS instance equivalents

| Goal | AWS instance | vCPU / RAM / GPU | On-demand $/h (us-east-1) | Spot $/h (typical) |
|------|--------------|------------------|--------------------------|--------------------|
| CPU smoke (= Azure D4s_v5) | `m6i.xlarge` | 4 / 16 / — | 0.192 | ~0.06 |
| CPU full (= Azure D16s_v5) | `m6i.4xlarge` | 16 / 64 / — | 0.768 | ~0.25 |
| T4 (= Azure NC4as_T4_v3) | `g4dn.xlarge` | 4 / 16 / 1× T4 16GB | 0.526 | ~0.16 |
| T4 bigger | `g4dn.2xlarge` | 8 / 32 / 1× T4 | 0.752 | ~0.23 |
| A10G (= Azure NV6ads_A10_v5) | `g5.xlarge` | 4 / 16 / 1× A10G 24GB | 1.006 | ~0.30 |
| A10G bigger | `g5.2xlarge` | 8 / 32 / 1× A10G | 1.212 | ~0.36 |
| L4 (newer, cheaper than A10G) | `g6.xlarge` | 4 / 16 / 1× L4 24GB | 0.805 | ~0.24 |
| V100 (legacy, often available) | `p3.2xlarge` | 8 / 61 / 1× V100 16GB | 3.06 | ~0.92 |

Prices as of late 2024; check current AWS pricing.

For Monty / habitat-sim the sweet spot is **g4dn.xlarge** (T4) or **g6.xlarge**
(L4). Both have ≥16 GB VRAM (enough for habitat-sim + Monty), and both run
the same `nvidia/cuda:12.1.1-runtime-ubuntu22.04` base image we already use.

## Quotas on AWS

AWS uses **Service Quotas** instead of Azure's per-family vCPU model. The
relevant quotas live under "EC2":

| Quota name | What it covers | Default for new account |
|------------|----------------|-------------------------|
| Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances | M6i, C6i, R6i… | 32 vCPU |
| Running On-Demand G and VT instances | g4dn, g5, g6 (T4/A10G/L4) | **0 vCPU** |
| Running On-Demand P instances | p3, p4, p5 (V100/A100/H100) | **0 vCPU** |
| All Spot Instance Requests | parallel spot capacity | usually 64 vCPU |

Request increases via `aws service-quotas request-service-quota-increase`
or the console. Approval for G/VT typically takes **a few hours to 2 days**
for low values (4–16 vCPU), 1–2 weeks for larger asks. Same shape as Azure;
neither cloud is faster than the other once you factor in human approval.

If you have an **existing** AWS account with billing history, the G/VT quota
is sometimes preset to 4 or 16 vCPU without asking. Worth checking:

```bash
aws service-quotas get-service-quota \
  --service-code ec2 \
  --quota-code L-DB2E81BA      # Running On-Demand G and VT
```

## Regions with the best chance of capacity

For G/L/T-series GPUs:

1. **us-east-1** (N. Virginia) — biggest pool, cheapest spot, but most
   contention.
2. **us-east-2** (Ohio) — cheaper spot than us-east-1 in practice.
3. **us-west-2** (Oregon) — large pool, often-better spot.
4. **eu-west-1** (Ireland) — best EU price/availability for G-series.
5. **eu-central-1** (Frankfurt) — closest to Sweden Central, slightly
   pricier, smaller pool.

## Migration effort from the current Azure stack

The current Azure stack is `Bicep + cloud-init + docker compose`. Mapping:

| Azure piece | AWS equivalent |
|-------------|----------------|
| `main.bicep` resource group | CloudFormation stack OR Terraform module |
| VM + NIC + NSG + public IP | `aws_instance` + security group + EIP |
| Managed identity + blob upload | IAM instance profile + S3 `aws s3 cp` |
| `cloud-init.yaml` `customData` | EC2 `user_data` (same cloud-init dialect, **works as-is**) |
| Storage account + container | S3 bucket |
| Quota survey (`find-quota.sh`) | `aws service-quotas list-service-quotas --service-code ec2` |
| `deploy.sh` | `terraform apply` or `aws cloudformation deploy` |

Estimated effort: **a few hours** to port `main.bicep` to a Terraform module
that takes the same parameters (`suite`, `vmSize`, `nEvalEpochs`,
`useSpot`). The cloud-init file does not need to change at all — AWS
EC2 supports the exact same `#cloud-config` format.

## Recommendation

1. **Run the smoke matrix on the Azure D4s_v5 we just deployed.** It's
   already provisioned and is the cheapest path to a real result. Nothing
   AWS offers is meaningfully cheaper for a 4 vCPU CPU box.
2. **In parallel, request quotas in both clouds** so a future GPU run
   isn't blocked again:
   - Azure: standardNCASv3_T4 (4 vCPU) in `swedencentral` and `eastus2`.
   - AWS: "Running On-Demand G and VT instances" (16 vCPU) in `us-east-2`
     and `eu-west-1`.
3. **For an immediate GPU run**, spin up a **Lambda Labs `gpu_1x_a10`**
   (or `gpu_1x_a10g`) and run `docker compose up` from the existing
   `docker/` folder. No quota wait. ~$0.75/h on-demand. Stop when done.
4. **For a recurring GPU benchmark cadence**, port the Bicep module to
   Terraform once and keep both Azure and AWS deployable from the same
   repo. EC2 spot for `g6.xlarge` lands around $0.24/h, which beats
   Azure spot NV6ads_A10_v5 in most regions.

There is no killer reason to abandon Azure here — AWS is a peer, not a
shortcut. The shortcut, if you need GPU minutes today, is Lambda Labs.
