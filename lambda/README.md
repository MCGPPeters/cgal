# Lambda Labs runner

One-shot deploy of the CGAL benchmark to a Lambda Labs GPU instance.
The Lambda image (`lambda-stack-22-04`) ships with NVIDIA drivers, CUDA,
Docker, and the NVIDIA container toolkit preinstalled, so bootstrap is
just `git clone && docker compose build && docker compose run`.

## One-time setup

1. Create an API key at <https://cloud.lambdalabs.com/api-keys>.
2. Export it (and put it in your `~/.zshrc` if you want it to stick):
   ```bash
   export LAMBDA_API_KEY='secret_...'
   ```
3. The script auto-uploads `~/.ssh/id_ed25519.pub` as SSH key
   `cgal-bench` the first time it runs. No SSH config needed.

## Run a smoke benchmark

```bash
cd lambda
./run.sh                     # SUITE=smoke INSTANCE_TYPE=gpu_1x_a10
```

The script will:
1. Pick a region with capacity for `gpu_1x_a10`.
2. Launch a 1× A10 instance (~$0.75/h).
3. SSH in, clone cgal + tbp.monty, build the monty Docker image.
4. Download YCB + pretrained models *inside* the container.
5. Run the smoke matrix (~20-40 min wall-clock on 1× A10).
6. `scp` `docker/logs/` back into `lambda/results-<timestamp>/`.
7. **Terminate the instance** so the meter stops.

Estimated cost for smoke: **~$0.30-0.50**.

## Variations

```bash
# Different GPU
INSTANCE_TYPE=gpu_1x_a6000 ./run.sh   # A6000 48GB, ~$0.80/h
INSTANCE_TYPE=gpu_1x_h100_pcie ./run.sh  # H100 80GB, ~$2.49/h

# Different region (defaults to first with capacity)
REGION=us-west-1 ./run.sh

# Full matrix instead of smoke (warning: hours, $$$ on H100)
SUITE=full ./run.sh

# Don't auto-terminate (you'll pay until you call teardown.sh)
KEEP_ALIVE=true ./run.sh

# Use a different fork / branch
CGAL_REF=my-branch MONTY_REF=my-monty-branch ./run.sh
```

## Manual teardown

If you set `KEEP_ALIVE=true` or the script crashed before termination:

```bash
./teardown.sh <instance_id>
# or just kill all your instances:
curl -u "$LAMBDA_API_KEY:" \
  https://cloud.lambdalabs.com/api/v1/instances \
  | jq -r '.data[].id' \
  | xargs -I{} ./teardown.sh {}
```

## Capacity caveat

Lambda Labs sells GPU capacity on demand and **runs out**. If `./run.sh`
prints "no capacity for gpu_1x_a10 right now", it lists every instance
type that *does* have capacity along with prices. Pick one and rerun:

```bash
INSTANCE_TYPE=gpu_1x_a6000 ./run.sh
```

A10/A6000 are usually available. H100 / 8× variants are routinely sold
out.
