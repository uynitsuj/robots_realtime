# xdof_hq_v8dj_recorded — lab42 "recorded gains" controller profile for the us05 YAMs

Mirror of lab42 market42's `gravity_comp_profile: "v8dj_recorded"` (the default for new YAM nodes there and the
US05 `v12dj_recent` policy baseline). Use these instead of `robot_configs/yam/xdof_hq/*.yaml` when a policy should
move the way it does under market42.

| setting (joints 1–7)          | xdof_hq (old)                      | xdof_hq_v8dj_recorded (= lab42)                    |
| ----------------------------- | ---------------------------------- | -------------------------------------------------- |
| gravity model                 | arm only (`yam.xml`, no gripper)   | arm + linear_4310 gripper (`yam_linear_4310_gravity.xml`) |
| gravity_comp_factor           | 1.45                               | 1.3                                                |
| kp                            | 80 80 80 **40 15 15** 20           | 80 80 80 **10 10 10** 20                           |
| kd                            | 5 5 5 1.5 1.5 1.5 0.5              | same                                               |
| idle damping / Coulomb ff     | none (this i2rt has none)          | zero / off                                         |
| joint limits                  | hand-tightened                     | i2rt YAM v1 ranges ± 0.15 rad (get_yam_robot)      |
| gripper                       | auto-calibrated, force limit 50    | same (kp 20 / kd 0.5)                              |

Gravity torques of the two models at the session startup pose (Nm, joints 2–4; `make_gravity_model.py` prints them):
arm+gripper ×1.3 → −5.3 / +5.9 / +2.1; arm-only ×1.45 → −3.3 / +3.8 / +0.9. The old profile leaves ~1.2 Nm on joint 4
to the PD loop, which is fine at kp 40 (0.03 rad) but would sag 0.12 rad at kp 10 — so the gripper mass has to be in
the model before the wrist gains can be lowered. The inertial-only XML matches lab42's meshed model to < 0.04 Nm.
Regenerate with `.venv/bin/python robot_configs/yam/xdof_hq_v8dj_recorded/make_gravity_model.py`.
