![github actions CI badge](https://github.com/lewandd/nelegolizer/actions/workflows/test.yml/badge.svg?branch=main)

# NeLegolizer (work in progress)

NeLegolizer is Python project that transform 3D object into a LEGO layout using neural networks.

## Getting Started

### Prerequisites
Make sure that you have installed latest version of pip and setuptools.
```
pip install --upgrade pip setuptools
```
### Installation

1. Clone the repo
   
   ```sh
   git clone https://github.com/lewandd/nelegolizer.git
   cd nelegolizer
   ```
3. Install required packages
   ```sh
   pip install -r requirements.txt --no-cache-dir
   ```
4. Install package with editable mode
   ```sh
   pip install -e nelegolizer
   ```

## Usage
Load 3d object and get list of lego bricks ([list of supported file extensions](https://docs.pyvista.org/api/readers/_autosummary/pyvista.get_reader.html#pyvista.get_reader)).
```python
from nelegolizer import legolize
from nelegolizer.data import LDrawModel, LDrawFile

lego_bricks = legolize("path/to/model_3d.obj")
```
Then you can use results by either:
- Save results as mpd file ([LDraw File Format Specification](https://www.ldraw.org/article/218.html))
  ```python
  ldraw_model = LDrawModel.from_bricks(lego_bricks, "Model Name")
  ldraw_file = LDrawFile()
  ldraw_file.add_model(ldraw_model)
  ldraw_file.save("legolized_cone.mpd")
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
