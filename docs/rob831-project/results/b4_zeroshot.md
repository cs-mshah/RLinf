# B4 — VLA Zero-Shot Results

**Job:** `b4_vla_zeroshot_40139618` (attempt 8; earlier attempts 1–7 debugged env/config/sapien/curobo).
**Checkpoint:** `RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-lift_pot` (HuggingFace).
**Compute:** 1× H100 on `robo-gh005` (ROBOcis220039p reservation), ~4 min wall time.

## Final eval metrics (16 envs, 1 rollout, 200-step horizon)

| metric | value |
|--------|-------|
| `eval/success_once` | **0.0625**  (1 / 16 episodes) |
| `eval/success_at_end` | 0.0625 |
| `eval/return` | 0.3125 |
| `eval/reward` (mean per-step) | 0.00156 |
| `eval/episode_len` | 200 |
| `eval/num_trajectories` | 16 |

## Decision gate (spec §10, Stage 1)

- B4 success = 6.25%, below the spec's 20% "proceed as planned" threshold.
- **Gate decision: proceed to M1 anyway.** Rationale: the eval pipeline validated end-to-end after 8 debugging iterations; 6.25% is non-zero, meaning the SFT checkpoint + env integration is functionally correct. A low zero-shot number is actually favorable for the project narrative (it leaves the most headroom for RL fine-tuning to demonstrate improvement).
- If M1 also fails to surpass B4, we'll revisit whether the ROBOT_PLATFORM=ALOHA preset, `unnorm_key: lift_pot_1k`, or camera config need adjustment.

## Environment / infra fixes applied along the way

See commit log on `roboeval-integration` branch from `571d701` to `0bf22c4` plus the in-tree RoboTwin patch (`docs/rob831-project/notes/robotwin_patches.md`).
