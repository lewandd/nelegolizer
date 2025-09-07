import numpy as np

def compute_bounds(lb_list):
        mins = []
        maxs = []
        
        for b in lb_list:
            b_min = np.array(b.position)
            b_max = b_min + np.array(b.rotated_shape)
            mins.append(b_min)
            maxs.append(b_max)
        
        mins = np.min(mins, axis=0)
        maxs = np.max(maxs, axis=0)
        
        return mins, maxs