# jlj_robot

这是 A00 机器人简单行走参考数据和仿真回放文件。

```bash
python3 generate_simple_walk_dataset.py --out simple_walk_a00.npz --seconds 10
pip install mujoco
python3 play_simple_walk_mujoco.py --xml A00_robot_mujoco_tuned.xml --data simple_walk_a00.npz
```
