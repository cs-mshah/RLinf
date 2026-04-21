#!/bin/bash
#SBATCH --job-name=diag_render
#SBATCH --partition=ROBO
#SBATCH --reservation=ROBOcis220039p
#SBATCH --nodelist=robo-gh005
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:10:00
#SBATCH -A cis220039p
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# Probe robo-gh005's rendering / Vulkan / sapien capabilities to diagnose
# the 'failed to find a rendering device' error seen in B4.

source /ocean/projects/cis220039p/hkwon/projects/RLinf/.venv/bin/activate

echo "=== hostname / nvidia-smi ==="
hostname
nvidia-smi -L 2>&1 | head
echo ""
echo "=== Vulkan ICD locations ==="
ls /usr/share/vulkan/icd.d/ 2>&1
ls /etc/vulkan/icd.d/ 2>&1 || true
echo ""
echo "=== nvidia-specific libs ==="
ls /usr/lib64/libnvidia* 2>&1 | head -5
find / -maxdepth 4 -name "nvidia_icd*json" 2>/dev/null | head
echo ""
echo "=== vulkaninfo (if available) ==="
which vulkaninfo 2>&1
vulkaninfo --summary 2>&1 | head -40 || true
echo ""
echo "=== sapien render probe ==="
python - <<'PY'
import os
print("VK_ICD_FILENAMES=", os.environ.get("VK_ICD_FILENAMES"))
print("SAPIEN env:", {k: v for k, v in os.environ.items() if k.startswith("SAPIEN")})
try:
    import sapien
    print("sapien version:", sapien.__version__)
    print("sapien file:", sapien.__file__)
    # Try creating a renderer.
    from sapien import Scene
    scene = Scene()
    scene.add_ground(0.0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    print("Scene created OK")
except Exception as e:
    print(f"sapien probe FAILED: {type(e).__name__}: {e}")
PY
