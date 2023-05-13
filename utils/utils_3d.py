
from pyntcloud import PyntCloud
import numpy as np
import pandas as pd

def generate_single_camera_ply(P,R,file_name,color=[0,255,0],is_swap_yz_for_threejs = True):

    if is_swap_yz_for_threejs:
        rot = np.array([[1,0,0],[0,0,1],[0,1,0]])
        P = rot.dot(P)
        R = rot.dot(R)

    vertices,edges  =make_view_cone_for_ply(P,R,0,f=1)
    
    """write header"""
    header = (
    'ply\n\
format ascii 1.0\n\
element vertex {}\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
element face {}\n\
property list uchar int vertex_indices\n\
end_header\n').format(len(vertices),len(edges)
)

    with open(file_name, 'w') as f:
        f.write(header)
        
    cols = np.tile(np.array(color),(vertices.shape[0],1))
    vertices_w_col = np.concatenate([vertices,cols], axis=1)
    
    with open(file_name, 'a') as f:
        np.savetxt(f, vertices_w_col, fmt='%.5f')

    with open(file_name, 'a') as f:
        np.savetxt(f, edges, fmt='%i')

def generate_multiple_camera_ply(Ps,Rs,file_name,color=[0,255,0],is_swap_yz_for_threejs = True):
    
    """write header"""
    header = (
    'ply\n\
format ascii 1.0\n\
element vertex {}\n\
property float x\n\
property float y\n\
property float z\n\
property uchar red\n\
property uchar green\n\
property uchar blue\n\
element face {}\n\
property list uchar int vertex_indices\n\
end_header\n').format(len(Ps)*5,len(Ps)*4
)

    with open(file_name, 'w') as f:
        f.write(header)

    vs = []
    es = []
    for idx,(P,R) in enumerate(zip(Ps,Rs),0):

        if is_swap_yz_for_threejs:
            rot = np.array([[1,0,0],[0,0,1],[0,1,0]])
            P = rot.dot(P)
            R = rot.dot(R)

        vertices,edges  =make_view_cone_for_ply(P,R,stat_vertex_idx=idx*5,f=0.2)

        cols = np.tile(np.array(color),(vertices.shape[0],1))
        vertices_w_col = np.concatenate([vertices,cols], axis=1)

        vs.append(vertices_w_col)
        es.append(edges)
    
    for v in vs:
        with open(file_name, 'a') as f:
            np.savetxt(f, v, fmt='%.5f')
            
    for e in es:
        with open(file_name, 'a') as f:
            np.savetxt(f, e, fmt='%i')

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
    
    vertices = np.array([pos,p1,p2,p3,p4])
    edges = [[3,stat_vertex_idx+0,stat_vertex_idx+1,stat_vertex_idx+2],
             [3,stat_vertex_idx+0,stat_vertex_idx+2,stat_vertex_idx+3],
             [3,stat_vertex_idx+0,stat_vertex_idx+3,stat_vertex_idx+4],
             [3,stat_vertex_idx+0,stat_vertex_idx+4,stat_vertex_idx+1]]
    return vertices,edges

def save_3dpoints_ply(pts,out_ply_file_name, is_write_text,is_swap_yz_for_threejs = True):
    # ref: https://pyntcloud.readthedocs.io/en/latest/io.html
    # the doc is scarce and not complete

    n = len(pts)

    # The points must be written as a dataframe,
    # ref: https://stackoverflow.com/q/70304087/6064933

    if is_swap_yz_for_threejs:
        x = pts[:, 0]
        y = pts[:, 2]
        z = pts[:, 1]
    else:
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

    data = {'x': x,
            'y': y,
            'z': z,
            'red': 255, #np.random.rand(n),
            'blue': 0,
            'green': 0
            }

    # build a cloud
    cloud = PyntCloud(pd.DataFrame(data))

    # the argument for writing ply file can be found in
    # https://github.com/daavoo/pyntcloud/blob/7dcf5441c3b9cec5bbbfb0c71be32728d74666fe/pyntcloud/io/ply.py#L173
    cloud.to_file(out_ply_file_name, as_text=is_write_text)

