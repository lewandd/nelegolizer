[![CI](https://github.com/lewandd/nelegolizer/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/lewandd/nelegolizer/actions/workflows/test.yml)

# NeLegolizer 🧱

## Overwiew 
**Nelegolizer** is a project focused on transforming 3D objects into LEGO-like models using neural networks. It voxelizes 3D meshes, analyzes them through convolutional neural networks, and generates consistent LEGO brick representations (including export in the LDraw format).

The name **nelegolizer** combines **neural networks** and **legolization**, where *legolization* is a term describing the process of converting data into LEGO-like representations.

## How It Differs from Other Approaches

Traditional LEGO builders rely on fixed rules or heuristic algorithms to decompose 3D shapes into LEGO bricks. Such approaches can produce reasonable results but fail to capture **context-dependent building styles** or **subtle design preferences** that human builders naturally apply.

**Nelegolizer** overcomes this limitation by using neural networks trained on LEGO constructions. This allows the system to learn complex patterns and preferences that are difficult — if not impossible — to define explicitly in code. For example, during training, the model may learn that in certain structural or aesthetic contexts, it is better to use two 2×1 bricks instead of a single 4×1 brick. While such a choice might not be optimal from a purely structural stability standpoint — which is often the only criterion considered in traditional methods — it may reflects realistic and stylistic building patterns found in real LEGO constructions.

In essence, **nelegolizer** learns **how LEGO builders think**, not just how to fill space with bricks — a capability that traditional algorithmic methods cannot replicate.

In addition: 
- **Nelegolizer** also supports non-rectangular bricks with various shapes and heights, unlike most existing approaches. 
- **Nelegolizer** is a **universal tool**, not limited to any specific 3D object category — unlike many recent ML-based legolization approaches that are often tailored to predefined model classes.
  
## Getting Started

### Prerequisites
Make sure that you have installed the latest version of pip and setuptools.
```
pip install --upgrade pip setuptools
```
### Installation
Clone the repository, install dependencies, and install the **nelegolizer** package in editable mode.
```sh
git clone https://github.com/lewandd/nelegolizer.git
cd nelegolizer
pip install -r requirements.txt
pip install -e nelegolizer
```

### Usage
Load a 3D object with a [supported file extension](https://docs.pyvista.org/api/readers/_autosummary/pyvista.get_reader.html#pyvista.get_reader) and get a list of LEGO bricks.

```python
from nelegolizer import legolize
from nelegolizer.data import LDrawModel, LDrawFile

lego_bricks = legolize("path/to/model_3d.obj")
```
Then you can use results by either:
- Save results as [MPD file](https://www.ldraw.org/article/218.html)
  ```python
  ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
  ldraw_file = LDrawFile()
  ldraw_file.add_model(ldraw_model)
  ldraw_file.save("legolized_model.mpd")
  ```
- Use bricks data in your application
  ```python
  for brick in lego_bricks:
    pos = brick.mesh_position
    rot = brick.rotation # along y axis
    col = brick.color
    part_id = brick.part.brick_id
    part_size = brick.part.size
    ...
  ```
- Visualize bricks
  ```python
  import pyvista as pv

  plotter = pv.Plotter()
  for brick in lego_bricks:
    plotter.add_mesh(brick.mesh)
  plotter.show()
  ```
## License

Distributed under the MIT License. See `LICENSE` for more information.
