---
name: gpu-kernel-module-missing
description: NVIDIA GPU diagnosis recipe; on this laptop the kernel module is often missing for the running kernel, not a userspace problem
metadata:
  type: project
---

This laptop (abhi-Thin-GF63-12UDX) has an RTX 3050 6GB Laptop GPU. When CUDA/cuDF/numba report no devices, the root cause is almost always **the NVIDIA kernel module not being built/installed for the currently running kernel**, not a CUDA toolkit or cuDF version mismatch.

**Why:** Ubuntu ships the 570 driver as prebuilt-per-kernel `linux-modules-nvidia-570-<kernel>` packages, and `dkms` is NOT installed here. When the kernel is upgraded (e.g. `6.17.0-22` -> `6.17.0-23`), the matching nvidia module package is often not pulled in automatically, so the new kernel boots without `nvidia.ko`. The userspace stack (`nvidia-driver-570`, `libcudart12`, `nvcc 12.0`, `cudf 25.04`, `numba-cuda`) is all in place and works fine -- only the kernel side is broken.

**How to apply:** When diagnosing GPU failures on this machine, check these signals **first** before touching cuDF/numba/CUDA versions:

1. `lsmod | grep nvidia` -- empty means module not loaded
2. `ls /dev/nvidia*` -- absent means kernel side is dead
3. `modprobe -n -v nvidia` -- `FATAL: Module nvidia not found in directory /lib/modules/$(uname -r)` confirms it
4. `find /lib/modules -name 'nvidia.ko*'` -- shows which kernels DO have the module built; you can boot back into one of those from GRUB as a stopgap

**The fix is one of:**
- `sudo apt install linux-modules-nvidia-570-$(uname -r)` (per-kernel)
- `sudo apt install dkms nvidia-dkms-570 && sudo dkms autoinstall` (permanent, auto-rebuilds on kernel upgrades)

**Secondary gotcha:** `CUDA_VISIBLE_DEVICES=''` is being exported in the harness shell environment (not found in `~/.bashrc`, `~/.profile`, `~/.zshrc`, `~/.zshenv`, `~/.config/environment.d/`, or `/etc/environment` — likely set by a launcher wrapper). Empty string means "hide all GPUs from CUDA". Even with a working kernel module, this would silently mask the GPU. Always `unset CUDA_VISIBLE_DEVICES` (or `export CUDA_VISIBLE_DEVICES=0`) when verifying a GPU fix here.

Related: [[prediction_lgbm_cpu]] -- prediction.py already falls back to CPU LightGBM on this box; same root cause.
