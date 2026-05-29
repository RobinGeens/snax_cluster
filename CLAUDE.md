# CLAUDE.md

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## 5. Building

All commands from `target/snitch_cluster/`

**Podman shorthand** (used for RTL gen and SW compilation):
```bash
# Mount the whole repo (compiles reference repo-root paths like sw/snRuntime), keep CWD at target/snitch_cluster.
ROOT=$(git rev-parse --show-toplevel)
POD="podman run --rm -i -v $ROOT:$ROOT -w $(pwd) ghcr.io/kuleuven-micas/snax:main"
```
Mounting only `$(pwd)` (target/snitch_cluster) fails: includes like `alloc_decls.h` live under the repo-root `sw/`, outside that subtree.

### Day-to-day: rebuild one app and simulate

After editing C source or `params_in.hjson`:
```bash
$POD make -C sw/apps/<app_name>              # rebuild single app (in container)
./bin/snitch_cluster.vsim sw/apps/<app_name>/build/<app_name>.elf |& tee tmp-<app_name>.log
```
This only recompiles the changed app (+ re-runs datagen if params changed). Takes seconds.

### Rebuild all SW

```bash
$POD make $CFG sw                            # rebuild all enabled apps (in container)
```

### Rebuild vsim (after unwanted `make clean`)

```bash
$POD make $CFG rtl-gen                       # regenerate RTL wrappers (in container)
$POD make $CFG vsim_preparation              # rebuild fesvr + bootdata (in container)
make $CFG bin/snitch_cluster.vsim            # recompile Questasim simulator (on host, needs license)
```

### Full build from scratch

Only needed on a fresh clone or after `make clean`:
```bash
# From repo root:
./scripts/build_sim.sh          # clean + RTL + SW + vsim (slow, uses podman)
```

**Active apps** are listed in `sw/apps/Makefile` (`SUBDIRS`). Commented-out entries are disabled.

## 6. Running Simulations

Always redirect to `tmp-<app_name>.log` so the output can be followed with `tail -f`:

```bash
./bin/snitch_cluster.vsim sw/apps/<app_name>/build/<app_name>.elf |& tee tmp-<app_name>.log
```

Don't append anything (like the parameter, current iteration, etc.) to the `tmp-<app_name>` filename. Prevent making many different log files for the same app.


- `exit code = 0` at the end of the log = PASS.
- Waveforms: `vsim.wlf`. Per-hart traces: `logs/trace_chip_*_hart_*.dasm`.

## 7. Changing Apps — Test Small, Debug Until Pass

**Whenever you change app source, params, or datagen: shrink parameters, run the sim, fix failures, repeat.**

Do not hand off untested changes or wait for the user to run sims. Loop locally the errors are reduced to quantization noise.

### 1. Use small parameters first

Edit `sw/apps/<app_name>/data/params_in.hjson` to the smallest meaningful case. Goal: fast rebuild + fast sim, not full benchmark size.

### 2. Rebuild and simulate after every meaningful change

```bash
$POD make -C sw/apps/<app_name>
./bin/snitch_cluster.vsim sw/apps/<app_name>/build/<app_name>.elf |& tee tmp-<app_name>.log
```

### 3. Debug on the go

On failure, inspect `tmp-<app_name>.log` (and traces under `logs/` if needed), fix the root cause, rebuild, and re-run. Repeat until PASS.

### 4. Done when verified

Success = small-params sim passes with your changes. Mention any param values you temporarily reduced so the user can scale up if needed.

## 8. chisel-ssm

Bender dependency declared in `Bender.yml`, local checkout at `/esat/micas-lapserv11/users/rgeens/chisel-ssm`. Resolved by `bender path chisel-ssm`.

**Contents:**
- Scala data generators producing golden test data per app — invoked via sbt from `sw/apps/common-datagen.mk`
- Generated data: `chisel-ssm/generated/data/<app_name>/`
- Memory layout docs: `chisel-ssm/docs/memory_layouts/`

**Datagen flow** (runs automatically during app SW build):
1. `common-datagen.mk` checks `.datagen_cache/` for cached sbt output (keyed on L, D params)
2. Cache miss → runs `sbt "test:runMain <GeneratorClass> <args>"` inside chisel-ssm
3. Python `datagen.py` reads sbt output + `params_in.hjson` → produces `data.h`

