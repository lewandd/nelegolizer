"""
Interactive visualization of lego bricks.

Plot lego bricks with position, brick type and rotation changeable at runtime.
"""
import pyvista as pv
import numpy as np

from nelegolizer import const
from nelegolizer.data import LegoBrick
from nelegolizer.data import part_by_label

class MyCustomRoutine:
    def __init__(self, mesh) -> None:
        self.output = mesh  # Expected PyVista mesh type
        # default parameters
        self.kwargs = {
            "mesh_position_x": 0,
            "mesh_position_y": 0,
            "mesh_position_z": 0,
            "label": 0,
            "rotation": 0
        }

    def __call__(self, param, value):
        self.kwargs[param] = value
        self.update()

    def update(self) -> None:
        # This is where you call your simulation
        result = LegoBrick(label=self.kwargs["label"], 
                           mesh_position=np.array((self.kwargs["mesh_position_x"],self.kwargs["mesh_position_y"],self.kwargs["mesh_position_z"])) * const.BRICK_UNIT_MESH_SHAPE, 
                           rotation=self.kwargs["rotation"]).mesh
        self.output.copy_from(result)

starting_mesh = LegoBrick(label=0, 
                          mesh_position=np.array((0,0,0)) * const.BRICK_UNIT_MESH_SHAPE, 
                          rotation=0).mesh
engine = MyCustomRoutine(starting_mesh)

p = pv.Plotter()
p.add_mesh(starting_mesh, show_edges=False)
p.add_slider_widget(
    callback=lambda value: engine("mesh_position_x", float(value)),
    rng=[-3, 3],
    value=0,
    title="X",
    pointa=(0.025, 0.4),
    pointb=(0.31, 0.4),
    style="modern",
)
p.add_slider_widget(
    callback=lambda value: engine("mesh_position_y", float(value)),
    rng=[-3, 3],
    value=0,
    title="Y",
    pointa=(0.025, 0.25),
    pointb=(0.31, 0.25),
    style="modern",
)
p.add_slider_widget(
    callback=lambda value: engine("mesh_position_z", float(value)),
    rng=[-3, 3],
    value=0,
    title="Z",
    pointa=(0.025, 0.1),
    pointb=(0.31, 0.1),
    style="modern",
)

p.add_text("Brick", position=[500.0, 65.0], font_size=14)
p.add_checkbox_button_widget(callback=lambda value: engine("label", int(0)),
                             color_on="grey",
                             position=(500.0, 3.0))
p.add_checkbox_button_widget(callback=lambda value: engine("label", int(1)),
                             color_on="grey",
                             position=(550.0, 3.0))
p.add_text(f"{part_by_label[0].brick_id}", position=[507.0, 20.0], font_size=8)
p.add_text(f"{part_by_label[1].brick_id}", position=[557.0, 20.0], font_size=8)

p.add_text("Rotation", position=[800.0, 60.0], font_size=14)
p.add_checkbox_button_widget(callback=lambda value: engine("rotation", int(0)),
                             color_on="grey",
                             position=(800.0, 3.0))
p.add_checkbox_button_widget(callback=lambda value: engine("rotation", int(90)),
                             color_on="grey",
                             position=(850.0, 3.0))
p.add_checkbox_button_widget(callback=lambda value: engine("rotation", int(180)),
                             color_on="grey",
                             position=(900.0, 3.0))
p.add_checkbox_button_widget(callback=lambda value: engine("rotation", int(270)),
                             color_on="grey",
                             position=(950.0, 3.0))
p.add_text("0", position=[820.0, 20.0], font_size=8)
p.add_text("90", position=[865.0, 20.0], font_size=8)
p.add_text("180", position=[910.0, 20.0], font_size=8)
p.add_text("270", position=[960.0, 20.0], font_size=8)


p.show_bounds()
p.show()