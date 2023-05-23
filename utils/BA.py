import numpy as np
import g2o

from collections import defaultdict

class BundleAdjust:
    
    def __init__(self,K):
        
        self.K = K
        
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverPCGSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(solver)
        
        focal_length =self.K[0,0]
        principal_point = (self.K[0,2], self.K[1,2])
        cam = g2o.CameraParameters(focal_length, principal_point, 0)
        cam.set_id(0)
        self.optimizer.add_parameter(cam)
        
    # --
    # Arges:
    # uvs  (m x n x 2) [np.ndarray]: m - number of views, n - number of points, projected 2D points of each view.
    # lmks (m x 3) [np.ndarray]: m - number of view, landmarks 3D cordinate with respect to world. 
    # Rs (m x 3 x 3) [np.ndarray]: m - number of views, attitude of each view.
    # Ps (m x 3 x 1) [np.ndarray]: m - number of views, position of each view.
    # Return:
    #
    # ---
    def make_graph(self,uvs,lmks,Rs,Ps):
        
        poses = []
        for i,(R,t) in enumerate(zip(Rs,Ps)):
            pose = g2o.SE3Quat(R, t)
            poses.append(pose)
            
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(i)
            v_se3.set_estimate(pose)
            if i < 1:
                v_se3.set_fixed(True)
            self.optimizer.add_vertex(v_se3)
            
        point_id = Rs.shape[0]
        inliers = dict()
        sse = defaultdict(float)

        #pts1_inliers,pts2_inliers,pts3_inliers,wLmks,_ = self.get3Viewcoresspondance(0,1,2,True)
        #print(pts1_inliers.shape,pts2_inliers.shape,pts3_inliers.shape)
        #match_pts = np.array([pts1_inliers,pts2_inliers,pts3_inliers]).transpose(1,0,2)
        match_pts = uvs.transpose(1,0,2)
        for i, (point,match_pt_views) in enumerate(zip(lmks,match_pts)):
            visible = []
            for j, (pose,match_pt) in enumerate(zip(poses,match_pt_views)):
                #z = cam.cam_map(pose * point)
                R = pose.matrix()[:3,:3]
                t = pose.matrix()[:3,3]
                XX = self.K.dot(R.T.dot((point-t)))
                u = XX[0]/XX[2]
                v = XX[1]/XX[2]
                if 0 <= u < self.K[0,2] * 2 and 0 <= v < self.K[1,2] * 2:
                    visible.append((j, [u,v],match_pt))
            if len(visible) < 2:
                continue

            vp = g2o.VertexPointXYZ()
            vp.set_id(point_id)
            vp.set_marginalized(True)
            vp.set_estimate(point)
            self.optimizer.add_vertex(vp)

            inlier = True
            for j, z, pt in visible:

                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, vp)
                edge.set_vertex(1, self.optimizer.vertex(j))
                edge.set_measurement(pt)
                edge.set_information(np.identity(2))

                edge.set_robust_kernel(g2o.RobustKernelHuber())

                edge.set_parameter_id(0, 0)
                self.optimizer.add_edge(edge)

            # if inlier:
            #     inliers[point_id] = i
            #     error = cam.cam_map(pose *vp.estimate()) -visible[i][2]
            #     sse[0] += np.sum(error ** 2)
            point_id += 1
            
        print("num vertices:", len(self.optimizer.vertices()))
        print("num edges:", len(self.optimizer.edges()))
        
    def run_optim(self,iter_num = 100):
        
        print("Performing full BA:")
        self.optimizer.initialize_optimization()
        self.optimizer.set_verbose(True)
        self.optimizer.optimize(iter_num)
        
        vertices = self.optimizer.vertices()

        Rs = []
        Ps = []

        for i in range(3):
            R = vertices[i].estimate().rotation().matrix().T
            t = -R.dot(vertices[i].estimate().translation())
            Rs.append(R)
            Ps.append(t)
            
        return Rs,Ps
        