import pyvista as pv


class Plotter:
    def __init__(self):
        self.plotter = pv.Plotter()

    def set_kwargs(self, default_kwargs, **custom_kwargs):
        if not custom_kwargs:
            kwargs = default_kwargs
        else:
            kwargs = custom_kwargs
            for key in default_kwargs:
                if key not in custom_kwargs:
                    kwargs[key] = default_kwargs[key]
        return kwargs

    def add_points(self, points: pv.pyvista_ndarray, **kwargs):
        kwargs = self.set_kwargs({'render_points_as_spheres': True,
                                  'color': 'white',
                                  'point_size': 10},
                                 **kwargs)
        self.plotter.add_mesh(points, **kwargs)

    def add_voxels(self, voxels: pv.UnstructuredGrid, **kwargs):
        kwargs = self.set_kwargs({'show_edges': True,
                                  'opacity': 1,
                                  'color': 'white'},
                                 **kwargs)
        self.plotter.add_mesh(voxels, **kwargs)

    def add_mesh(self, mesh: pv.PolyData, **kwargs):
        kwargs = self.set_kwargs({},
                                 **kwargs)
        self.plotter.add_mesh(mesh, **kwargs)

    def add_points_labels(self, points: pv.PolyData, **kwargs):
        labels = [f"{[round(p[0], 3), round(p[1], 3), round(p[2], 3)]}"
                  for p in points]
        kwargs = self.set_kwargs({'always_visible': False,
                                  'labels': labels},
                                 **kwargs)
        self.plotter.add_point_labels(points, kwargs)

    def show_bounds(self, **kwargs):
        kwargs = self.set_kwargs({},
                                 **kwargs)
        self.plotter.show_bounds(**kwargs)

    def show(self, **kwargs):
        kwargs = self.set_kwargs({'cpos': 'iso'},
                                 **kwargs)
        self.plotter.show(**kwargs)
