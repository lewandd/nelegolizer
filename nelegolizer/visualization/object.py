import pyvista as pv

def get_plotter():
    return Plotter()

class Plotter:
    """Print 3D structure

    Available structures to plot: mesh, voxels, points

    Attributes:
        plotter (pyvista.Plotter) : pyvista plotter object 
        default_kwargs_points (dictionary) : default arguments for plotting points
        default_kwargs_voxels (dictionary) : default arguments for plotting voxels
        default_kwargs_mesh (dictionary) : default arguments for plotting mesh
        default_kwargs_points_labels (dictionary) : default arguments for plotting labels
        default_kwargs_show_bounds (dictionary) : default arguments for show_bounds method
        default_kwargs_show (dictionary) : default arguments for show method
        
    """
    
    def __init__(self):
        """Create pyvista plotter object and prepare default data"""
        
        self.plotter = pv.Plotter()
        # set default values for plotter
        self.default_kwargs_points = {'render_points_as_spheres': True, 
                                    'color': 'white', 
                                    'point_size': 10}
        self.default_kwargs_voxels = {'show_edges' : True, 
                                    'opacity': 0.7}
        self.default_kwargs_mesh = {}
        self.default_kwargs_points_labels = {'always_visible': False}
        self.default_kwargs_show_bounds = {}
        self.default_kwargs_show = {'cpos': 'iso'}

    def add_points(self, points, **kwargs):
        """Add points mesh to plot
        
        Args:
            points (pyvista.pyvista_ndarray) : points to plot
            kwargs (**kwargs) : non-default arguments for pyvista.plotter add_mesh method
        
        """
        if not kwargs:
            kwargs = self.default_kwargs_points
        else: 
            for key in self.default_kwargs_points:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_points[key]
        self.plotter.add_mesh(points, **kwargs)

    def add_voxels(self, voxels, **kwargs):
        """Add voxels mesh to plot
        
        Args:
            voxels (pyvista.UnstructuredGrid) : voxels to plot
            kwargs (**kwargs) : non-default arguments for pyvista.plotter add_mesh method
        
        """
        
        if not kwargs:
            kwargs = self.default_kwargs_voxels
        else: 
            for key in self.default_kwargs_voxels:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_voxels[key]
        self.plotter.add_mesh(voxels, **kwargs)
        
    def add_mesh(self, mesh, **kwargs):
        """Add mesh to plot
        
        Args:
            mesh (pyvista.PolyData) : mesh to plot
            kwargs (**kwargs) : non-default arguments for pyvista.plotter add_mesh method
        
        """
        if not kwargs:
            kwargs = self.default_kwargs_mesh
        else: 
            for key in self.default_kwargs_mesh:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_mesh[key]
        self.plotter.add_mesh(mesh, **kwargs)

    def add_points_labels(self, points, **kwargs):
        """Add points labels
        
        Args:
            points (pyvista.PolyData) : points to label
            kwargs (**kwargs) : non-default arguments for pyvista.plotter add_point_labels method
        
        """
        # default labels
        self.default_kwargs_points_labels['labels': [f"{[round(p[0], 3), round(p[1], 3), round(p[2], 3)]}" for p in points]]

        if not kwargs:
            kwargs = self.default_kwargs_points_labels
        else: 
            for key in self.default_kwargs_points_labels:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_points_labels[key]
        self.plotter.add_point_labels(points, kwargs)

    def show_bounds(self, **kwargs):
        """Show bounds

        Args:
            kwargs (**kwargs) : non-default arguments for pyvista.plotter show_bounds method
        
        """
        if not kwargs:
            kwargs = self.default_kwargs_show_bounds
        else: 
            for key in self.default_kwargs_show_bounds:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_show_bounds[key]
        self.plotter.show_bounds(**kwargs)

    def show(self, **kwargs):
        """Show

        Args:
            kwargs (**kwargs) : non-default arguments for pyvista.plotter show method
        
        """
        if not kwargs:
            kwargs = self.default_kwargs_show
        else: 
            for key in self.default_kwargs_show:
                if key not in kwargs:
                    kwargs[key] = self.default_kwargs_show[key]

        self.plotter.show(**kwargs)
    
    