
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd

def make_view_cone_for_ply(pos,rot,stat_vertex_idx=0,f=0.1,fov_deg=90,aspect_ratio_h2v=0.8):
    r1 = rot.T[0,:]
    r2 = rot.T[1,:]
    r3 = rot.T[2,:]
    
    half_fov_rad = fov_deg/180 * np.pi * 0.5
    bottom_offset = f * np.tan(half_fov_rad)
    
    p1 = pos + f*r3 + bottom_offset *(r1+ aspect_ratio_h2v * r2)
    p2 = pos + f*r3 + bottom_offset *(r1- aspect_ratio_h2v * r2)
    p3 = pos + f*r3 + bottom_offset *(-r1-aspect_ratio_h2v * r2)
    p4 = pos + f*r3 + bottom_offset *(-r1+aspect_ratio_h2v * r2)

    print(p1)
    
    vertices = np.array([pos,p1,p2,p3,p4])
    edges = [[3,stat_vertex_idx+0,stat_vertex_idx+1,stat_vertex_idx+2],
             [3,stat_vertex_idx+0,stat_vertex_idx+2,stat_vertex_idx+3],
             [3,stat_vertex_idx+0,stat_vertex_idx+3,stat_vertex_idx+4],
             [3,stat_vertex_idx+0,stat_vertex_idx+4,stat_vertex_idx+1]]
    return vertices,edges

def save_3dpoints_ply(pts,out_ply_file_name, is_write_text):
    # ref: https://pyntcloud.readthedocs.io/en/latest/io.html
    # the doc is scarce and not complete

    n = len(pts)

    # The points must be written as a dataframe,
    # ref: https://stackoverflow.com/q/70304087/6064933
    data = {'x': pts[:, 0],
            'y': pts[:, 1],
            'z': pts[:, 2],
            'red': 255, #np.random.rand(n),
            'blue': np.random.rand(n),
            'green': np.random.rand(n)
            }

    # build a cloud
    cloud = PyntCloud(pd.DataFrame(data))

    # the argument for writing ply file can be found in
    # https://github.com/daavoo/pyntcloud/blob/7dcf5441c3b9cec5bbbfb0c71be32728d74666fe/pyntcloud/io/ply.py#L173
    cloud.to_file(out_ply_file_name, as_text=is_write_text)

