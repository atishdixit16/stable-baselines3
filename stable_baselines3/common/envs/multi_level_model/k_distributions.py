import numpy as np
import gstools as gs

def get_channel_end_indices(nx=32, ny=32, lx=1.0, ly=1.0, channel_width=0.125, seed=1):
    assert channel_width<1.0 and channel_width>0.0, 'invalid channel width. condition violated: 0 < channel_width < 1'
    channel_left_end = np.random.uniform(0,(1.0-channel_width))
    channel_right_end = np.random.uniform(0,(1.0-channel_width))
    return channel_left_end, channel_right_end

def single_generate(nx=32,ny=32,lx=1.0,ly=1.0,channel_k=1.0, base_k=0.01, channel_width=0.125, channel_left_end=0.4375, channel_right_end=0.4375):
    index_left = round(channel_left_end*ny)
    index_right = round(channel_right_end*ny)
    grid_channel_width = round(channel_width*ny)
    k = base_k*np.ones((nx,ny))
    for i in range(nx):
        j = ( (index_right - index_left) / nx ) *i + index_left
        for w in range(grid_channel_width ):
            k[round(j)+w, i] = channel_k
    return k

def batch_generate(nx=32, ny=32, lx=1.0, ly=1.0, channel_k=1.0, base_k=0.01, channel_width_range=(0.1,0.3), sample_size=10, seed=1):
    np.random.seed(seed) #for reproducibility
    k_batch = []
    for _ in range(sample_size):
        channel_width = np.random.uniform(channel_width_range[0], channel_width_range[1]) 
        channel_left_end, channel_right_end = get_channel_end_indices(nx, ny, lx, ly, channel_width, seed)
        k = single_generate(nx,ny,lx,ly,channel_k, base_k, channel_width, channel_left_end, channel_right_end)
        k_batch.append(k)
    return np.array(k_batch)

def batch_generate_krige(nx, ny, lx, ly,
                         variance,
                         len_scale,
                         cond_pos,
                         cond_val,
                         angle,
                         n_samples,
                         seed):
    
    '''
    nx, ny, lx, ly: grid dicretization  (nx and ny) and length (lx and ly) in x and y directions  
    variance: variance value in exponential variagram (\sigma in equation 8)
    len_scale: length scale for the exponential variogram (l_1 and l_2 in equation 9)
    cond_pos: an array of positions where distribution values are known
    cond_val: an array of values corresponding to 'cond_pos' locations
    angle: rotation angle of the generated field
    n_samples: number of samples to be generated
    seed: random state value for reproducibility
    
    '''
    
    # generate variogram model
    model = gs.Exponential(dim=2, var=variance, len_scale=len_scale, angles=angle)
    
    # ordinary kriging
    krige = gs.krige.Ordinary(model, cond_pos, cond_val)
    srf = gs.CondSRF(krige)
    
    # generate samples
    step_x, step_y = lx/(nx-1), ly/(ny-1)
    g_cols, g_rows = np.arange(0,lx+1,step_x), np.arange(0,ly+1,step_y) # x and y directions correspond to matrix col and row 
    ks = []
    for i in range(n_samples): 
        k_ = srf.structured([g_rows, g_cols], seed=seed+i)
        ks.append(k_)
    ks = np.array(ks)
    
    return ks

    