import numpy as np
from numba import jit

@jit(nopython=True)
def hmean(x):
    return (np.sum(x**-1)/x.shape[0])**-1

@jit(nopython=True)
def mean_(x):
    return np.mean(x)

@jit(nopython=True)
def sum_(x):
    return np.sum(x)

@jit(nopython=True)
def get_partition_ind(fine_grid_nx, fine_grid_ny, coarse_grid_nx, coarse_grid_ny):
    
    '''
    generate partition indices to be used for upscaling
    
    ''' 
    
    dx, dy = int( fine_grid_nx/coarse_grid_nx ) , int(fine_grid_ny/coarse_grid_ny)
    p_1 = [i*dx for i in range(coarse_grid_nx)]; p_1.append(fine_grid_nx)
    p_0 = [i*dy for i in range(coarse_grid_ny)]; p_0.append(fine_grid_ny) 
    
    return (p_0, p_1)

@jit(nopython=True)
def fine_to_coarse_mapping(fine_array, partition_ind, func):
    
    if func=='mean':
        func_=mean_
    elif func=='hmean':
        func_=hmean
    else:
        func_=sum_
    
    p_0, p_1 = partition_ind[0], partition_ind[1]
    coarse_array = np.empty((len(p_0)-1, len(p_1)-1))
    for i in range(len(p_0)-1):
        for j in range(len(p_1)-1):
            coarse_array[i,j] = func_( np.ascontiguousarray(fine_array[p_0[i]:p_0[i+1], p_1[j]:p_1[j+1]]).reshape(-1) )
    return coarse_array
            
    
@jit(nopython=True)
def coarse_to_fine_mapping(coarse_array, partition_ind):
    p_0, p_1 = partition_ind[0], partition_ind[1]
    fine_array = np.empty((p_0[-1], p_1[-1]))
    for i in range(len(p_0)-1):
        for j in range(len(p_1)-1):
            fine_array[p_0[i]:p_0[i+1], p_1[j]:p_1[j+1]] = coarse_array[i,j]
    return fine_array