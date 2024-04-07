def get_scalars(voxels, labels, res):
    """ Return list of scalars in order corresponding to cells order
    
    Args:
        voxels (pyvista.UnstructuredGrid) : grid of cells
        labels (list) : grid of shape (res, res, res) assigning labels for locations
        res (int) : resolution of a grid

    Returns:
        list: list of scalars (floats) in order corresponding to cells order
    """

    scalars = []
    for i in range(voxels.GetNumberOfCells()):
        cell = voxels.GetCell(i)
        x, y, z = (cell.GetBounds()[0]/res, cell.GetBounds()[2]/res, cell.GetBounds()[4]/res)
        label = labels[int(x)][int(y)][int(z)]
        for j in range(8):
            scalars.append(label)
    return scalars