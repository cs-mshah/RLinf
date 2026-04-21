# RoboTwin Patches Applied Locally (not in our repo)

We patched one file in the locally-cloned RoboTwin repo (`$ROBOTWIN_PATH` =
`~/OCEANDIR/projects/RoboTwin`, branch `RLinf_support`) to work around a
curobo API drift. These patches live outside this repo; record them here for
reproducibility on a fresh RoboTwin clone.

## Patch 1 — `envs/robot/robot.py:137`

**Why:** `nvidia-curobo` installed by `requirements/install.sh` (HEAD of
NVlabs/curobo) reorganized its module layout; `curobo.types.math` is no longer
a package, only a flat `curobo/types.py` module. RoboTwin's `envs/robot/planner.py`
does `from curobo.types.math import Pose as CuroboPose` at module import, so
the `CuroboPlanner` class never gets defined and falls back to `CuroboPlanner = None`.
A later `isinstance(x, CuroboPlanner)` in `robot.py:137` then raises
`TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`.

Our config uses `planner_backend: mplib` so we don't need curobo at all — just
need to guard the `isinstance` check.

```diff
--- a/envs/robot/robot.py
+++ b/envs/robot/robot.py
@@ line 137 @@
-            if not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):
+            if CuroboPlanner is None:
+                # Patched for RLinf/PSC: curobo import failed (API drift with
+                # installed nvidia-curobo version). Fall back to rebuilding the
+                # planner each reset; user must be on planner_backend=mplib.
+                self.set_planner(scene=scene)
+            elif not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):
                 self.set_planner(scene=scene)
```

## How to re-apply on a fresh clone

```bash
cd $ROBOTWIN_PATH
# manually apply the diff above, or use `git apply` with a saved patch file.
```

If upstream RoboTwin later pins a compatible curobo version, this patch can
be dropped.
