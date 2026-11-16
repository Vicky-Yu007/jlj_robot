# Model asset installation

The A00 mesh archive is intentionally kept outside normal text commits because STL and texture assets are binary. From the repository root, unpack the supplied archive into `assets/`:

```bash
mkdir -p assets
 tar -xzf A00_MuJoCo_XML.tar.gz -C /tmp
cp -a /tmp/A00_MuJoCo_XML/A00_robot_mujoco/meshes assets/
cp -a /tmp/A00_MuJoCo_XML/A00_robot_mujoco/textures assets/ 2>/dev/null || true
```

The corrected model is `assets/A00_robot_mujoco_supermimic.xml` and expects `assets/meshes/`.

After unpacking, check that all mesh files referenced by the XML exist:

```bash
python scripts/check_model_assets.py assets/A00_robot_mujoco_supermimic.xml
```
