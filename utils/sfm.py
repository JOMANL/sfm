from enum import IntEnum
import cv2
import g2o
import os
from numpy.linalg import svd, matrix_rank

import copy
import numpy as np
import random
from tmp import *

class InlierFlag(IntEnum):
    TriadOutlier = -3
    EpipolarOutlier = -2
    MatchingOutlier = -1
    Inlier = 1

class ViewBase:
    def __init__(self):
        self.im_gray = None
        self.key_point_uvs = None
        self.key_point_descriptors = None
        
        self.R = None
        self.t = None
        
        self.matching_inliers = None
        
    def set_image(self,im):
        self.im_gray = im
        
    def set_keypoint_and_descriptor(self,kp,des):
        self.key_point_uvs = kp
        self.key_point_descriptors = des

class Matcher:
    def __init__(self):
        self.detector = cv2.SIFT_create()
        self.kp_matcher = cv2.BFMatcher()
    
class Sfm3view:
    
    def __init__(self,wRv0,wPv0):
        
        # sample dataset directory
        sample_data_path = "../ImageDataset_SceauxCastle/images/"

        K = np.loadtxt(os.path.join(sample_data_path,"K.txt"))
        
        self.wLmks = None
        self.K = K
        self.invK = np.linalg.inv(K)
        
        self.F = None

        # For calculation Essential Matrix
        im1 = cv2.imread(os.path.join(sample_data_path,"100_7101.JPG"))
        im2 = cv2.imread(os.path.join(sample_data_path,"100_7102.JPG"))
        # For Pnp
        im3 = cv2.imread(os.path.join(sample_data_path,"100_7103.JPG"))
        
        # gray
        im1_gray = cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY)
        im3_gray = cv2.cvtColor(im3,cv2.COLOR_RGB2GRAY)
        
        self.views = [ViewBase(),ViewBase(),ViewBase()]

        self.views[0].set_image(im1_gray)
        self.views[1].set_image(im2_gray)
        self.views[2].set_image(im3_gray)
        
        self.views[0].R = wRv0
        self.views[0].t = wPv0
        
        self.match_manage_table = []

        self.matcher = Matcher()
        
    def updateView(self,idx,view):
        self.views[idx] = view
        
    def detectKeyPoints(self,idx):
                
        kp, des = self.matcher.detector.detectAndCompute(self.views[idx].im_gray,None)
        
        self.views[idx].set_keypoint_and_descriptor(kp,des)
        
    def matching2views(self,idx1,idx2,th_ratio = 0.5):
        
        v1 = self.views[idx1]
        v2 = self.views[idx2]
        
        if(v1.key_point_uvs == None or v2.key_point_uvs == None):
            raise Exception("no key points are detected.")
        
        matches = self.matcher.kp_matcher.knnMatch(v1.key_point_descriptors,v2.key_point_descriptors, k=2)

        good = []
        for match1,match2 in matches:
            if match1.distance < th_ratio*match2.distance:
                good.append(match1)

        
        if len(self.match_manage_table) == 0:   
               
            for i,m in enumerate(good):
                # create new matching table
                    self.match_manage_table.append(
                        {
                            "ID":i,
                            "V0":m.queryIdx,
                            "V1":m.trainIdx,
                            "V2":None,
                            "wLmk":None,
                            "inlier": InlierFlag.Inlier
                        }
                    )
        # append pre existed table  
        else:
            tmp = []
            key_1st = "V{:d}".format(idx1)
            key_2nd = "V{:d}".format(idx2)  
            for i,m in enumerate(good):
                for dd in self.match_manage_table:
                    view1_array_idx = dd[key_1st]
                    
                    if view1_array_idx == m.queryIdx:
                        dd[key_2nd] = m.trainIdx
                    else:            
                        tmp.append(
                            {
                                "ID":i,
                                "V0":None,
                                "V1":m.queryIdx,
                                "V2":m.trainIdx,
                                "wLmk":None,
                                "inlier": InlierFlag.Inlier
                            }
                        )
                        
        if len(self.match_manage_table) == 0:
            self.match_manage_table.extend(tmp)
            
        return good
                
    def get2D_2Dcoresspondance(self,idx1,idx2,isFliterInlier = False):
        
        v1 = self.views[idx1]
        v2 = self.views[idx2]
        
        key_1st = "V{:d}".format(idx1)
        key_2nd = "V{:d}".format(idx2) 

        pts1 = []
        pts2 = []
        GIDs = []
        
        n = len(self.match_manage_table)
        for i in range(n):
            
            view1_array_idx = self.match_manage_table[i][key_1st]
            view2_array_idx = self.match_manage_table[i][key_2nd]
            
            if (view1_array_idx is not None and view2_array_idx is not None):
                
                if isFliterInlier:
                    if (self.match_manage_table[i]["inlier"] == InlierFlag.Inlier):

                        pts1.append(v1.key_point_uvs[view1_array_idx].pt)
                        pts2.append(v2.key_point_uvs[view2_array_idx].pt)
                        GIDs.append(self.match_manage_table[i]["ID"])

                else:
                    print("ddd")
                    pts1.append(v1.key_point_uvs[view1_array_idx].pt)
                    pts2.append(v2.key_point_uvs[view2_array_idx].pt)
                    GIDs.append(self.match_manage_table[i]["ID"])
        
        self.updateView(idx1,v1)
        self.updateView(idx2,v2)
        
        return pts1,pts2,GIDs
    
    def get2D_3Dcoresspondance(self,idx1,idx2,isFliterInlier = False):
        
        view2 = self.views[idx2]

        key_1st = "V{:d}".format(idx1)
        key_2nd = "V{:d}".format(idx2) 

        pts2 = []
        lmks = []
        gids = []

        for dd in self.match_manage_table:
            
            view1_array_idx = dd[key_1st]
            view2_array_idx = dd[key_2nd]
            
            if view1_array_idx is not None and view2_array_idx is not None and dd["wLmk"] is not None:
                if isFliterInlier:
                    if dd["inlier"] == InlierFlag.Inlier:
                        pts2.append(view2.key_point_uvs[view2_array_idx].pt)
                        lmks.append(dd["wLmk"])
                        gids.append(dd["ID"])
                else:
                    pts2.append(view2.key_point_uvs[view2_array_idx].pt)
                    lmks.append(dd["wLmk"])
                    gids.append(dd["ID"])          
                
        pts2 = np.array(pts2).astype(np.float64)
        lmks = np.array(lmks).astype(np.float64)

        return pts2,lmks,gids
    
    def solveEpipolarEqToGetTranslation_2views(self,idx1,idx2):

        v1 = self.views[idx1]
        v2 = self.views[idx2]
        
        pts1,pts2,gids = self.get2D_2Dcoresspondance(idx1,idx2,True)
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        self.F, epipolar_inlier = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        epipolar_inlier_mask = epipolar_inlier.ravel()
        print(len(epipolar_inlier_mask))
               
        for i,(gid,inlier_flag) in enumerate(zip(gids,epipolar_inlier_mask)):
            AssertionError(i == gid)
            
            self.match_manage_table[gid]["inlier"] = InlierFlag.Inlier if inlier_flag == 1 else InlierFlag.EpipolarOutlier
        
        self.updateView(idx1,v1)
        self.updateView(idx2,v2)
        
    def solvePNP_byDLT(self,lmks,uvs,invK):
        n = lmks.shape[0]
        if uvs.shape[0] != n:
            raise  ValueError("[Error]")
        
        M = np.zeros((3*n,12+n))
        for i in range(n):
            M[3*i,0:3] = lmks[i,:]
            M[3*i,3] = 1
            M[3*i+1,4:7] = lmks[i,:]
            M[3*i+1,7] = 1
            M[3*i+2,8:11] = lmks[i,:]
            M[3*i+2,11] = 1
            M[3*i:3*i+3,i+12] = -np.array([uvs[i,0],uvs[i,1],1])
            
        U,S,V = np.linalg.svd(M)
        
        P = V[-1,:12].reshape(3,4)
        R = P[:3,:3]
        t = P[:3,3]
        u, _, v = np.linalg.svd(R)

        R = np.matmul(u, v)
        
        R = invK.dot(R)
        t = invK.dot(t)    

        d = np.identity(3)

        d[2][2] = np.linalg.det(np.matmul(u, v))

        R = np.dot(np.dot(u, d), v)
        C = -np.dot(R.T, t)

        if np.linalg.det(R) < 0:
            R = -R
            C = -C
        
        return R,C        
        
        
    def PnPRANSAC(self,X, x, calibration_matrix):

        new_r = np.identity(3)
        threshold = 5
        cnt = 0
        new_c = np.zeros((3, 1))
        homogenous_x = convertHomogeneouos(x)

        for i in range(500):
            ran_idx = random.sample(range(x.shape[0]), 6)

            alpha =X[ran_idx][:]

            beta =  x[ran_idx][:]

            C, R = LinearPnP(alpha, beta, calibration_matrix)
            S = []

            homo_alpha = homogenous_x[i][:]

            for j in range(x.shape[0]):
                reproj = proj3Dto2D(homo_alpha, calibration_matrix, C, R)

                e = np.sqrt(np.square((homogenous_x[j, 0]) - reproj[0]) + np.square((homogenous_x[j,1] - reproj[1])))

                if e < threshold:
                    S.append(j)

            countS = len(S)

            if (countS == x.shape[0]):
                break

            if (cnt < countS):
                cnt = countS
                new_r = R
                new_c = C

        return new_r, new_c
        
    def solvePNPToGetToGetTranslation_V2andV3(self,idx1,idx2):
        
        # check wheter 1st view's pose has been solved previously.
        if self.views[idx1].R is None or self.views[idx1].t is None:
            raise Exception("[Error] 1st view's pose must be solved previously.")
        
        pts,wLmks,_ = self.get2D_3Dcoresspondance(idx1,idx2,True)
        
        if pts.shape[0] == 0 or wLmks.shape[0] == 0:
            raise Exception("size of pts and wLmks must be > 0")
        
        if pts.shape[0] != wLmks.shape[0]:
            raise Exception("shape must be same")
        
        init_R = self.views[idx2].R
        trans_vec = self.views[idx2].t
        
        rot_vec,_ = cv2.Rodrigues(init_R)
        
        # (success, rot_vec, trans_vec,inliers) = cv2.solvePnPRansac(wLmks, pts.astype(np.float64), self.K.astype(np.double), 
        #                                                            np.zeros((5, 1)),
        #                                                            rot_vec,trans_vec,
        #                                                             flags=cv2.SOLVEPNP_P3P)
        
        # (success, rot_vec, trans_vec) = cv2.solvePnP(
        #     wLmks.astype(np.float64)[:8], pts.astype(np.float64)[:8], self.K.astype(np.double), 
        #                                                         np.zeros((5, 1)),
        #                                                         rot_vec,trans_vec,
        #                                                         flags=cv2.SOLVEPNP_ITERATIVE,useExtrinsicGuess = True)
        
        #wRv3,wPv3 = self.PnPRANSAC(wLmks,pts,self.K)#self.solvePNP_byDLT(wLmks,pts)
        
        #wRv3,wPv3 = self.solvePNP_byDLT(wLmks,pts,np.linalg.inv(self.K))
        
        # if not success:
        #     raise Exception("[Error] Fail to PNP")
        #wRv3, jacob = cv2.Rodrigues(rot_vec)
        #wPv3 = trans_vec.T[0]        
        
        wPv3 = trans_vec
        wRv3 = init_R
        
        return wPv3,wRv3
        
    def traiangulate_point(self,R1,t1,R2,t2,pt1,pt2):

        A = np.zeros((6,6))
        
        p1 = self.invK.dot(np.array([pt1[0],pt1[1],1]))
        p2 = self.invK.dot(np.array([pt2[0],pt2[1],1]))
        
        A[0:3,:3] = R1.T
        A[0:3,3] = -R1.T.dot(t1)
        A[3:6,:3] = R2.T
        A[3:6,3] = -R2.T.dot(t2)
        A[0:3,4] = -p1
        A[3:6,5] = -p2

        U,S,Vh = np.linalg.svd(A)
        X = Vh[-1,:4]

        a = (X / X[3])[:3]
        
        return a
        
    def decomposeE2Rt_and_calc_Xs_wrt_v1(self,idx1,idx2):
        
        # check wheter 1st view's pose has been solved previously.
        if self.views[idx1].R is None or self.views[idx1].t is None:
            raise Exception("[Error] 1st view's pose must be solved previously.")
        
        pts1,pts2,gids = self.get2D_2Dcoresspondance(idx1,idx2,True)
        pts1 = np.array(pts1).astype(np.int32)
        pts2 = np.array(pts2).astype(np.int32)
        print(pts1.shape)
          
        E = self.K.T.dot(self.F).dot(self.K)
        
        U,D,Vh = svd(E)
        
        # |R| = 1 is required to rotation matrix. Otherwise R = -R,t=-t is correct transformation.
        if np.linalg.det(U.dot(Vh))<0:
            Vh = -Vh
        
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        
        candidates_v1Pv2 = np.zeros((4,3)) # v2_v1Pv2
        candidates_v2Rv1 = np.zeros((4,3,3)) # v1Rv2

        candidates_v1Pv2[0] = U.T[2,:]
        candidates_v2Rv1[0] = U.dot(W).dot(Vh)

        candidates_v1Pv2[1] = -U.T[2,:]
        candidates_v2Rv1[1] = U.dot(W).dot(Vh)

        candidates_v1Pv2[2] = U.T[2,:]
        candidates_v2Rv1[2] = U.dot(W.T).dot(Vh)

        candidates_v1Pv2[3] = -U.T[2,:]
        candidates_v2Rv1[3] = U.dot(W.T).dot(Vh)

        # find correct geometory of R and t and execute traiangulation
        triad_positive_depth_counts = np.zeros(4)
        triad_inliers_masks = np.zeros((4,pts1.shape[0]))
        triad_v1Xs = []
        
        for cand_idx,(cand_v1Pv2,cand_v2Rv1) in enumerate(zip(candidates_v1Pv2,candidates_v2Rv1 )):
            
            v1Xs = []
            
            for i,(pt1,pt2) in enumerate(zip(pts1,pts2)):
                wRv1 = np.eye(3)
                wPv1 = np.zeros(3)
                wRv2 = wRv1.dot(cand_v2Rv1.T)
                wPv2 = wPv1 + cand_v1Pv2

                v1X = self.traiangulate_point(wRv1,wPv1,wRv2,wPv2,pt1,pt2)

                # check the sign of projected vector along camera axi
                flag = v1X[2] > 0 and cand_v2Rv1[:,2].dot(v1X - cand_v1Pv2) > 0
                if flag:
                    v1Xs.append(v1X)
                    triad_inliers_masks[cand_idx,i] = 1
                
            triad_positive_depth_counts[cand_idx] = np.count_nonzero(triad_inliers_masks[cand_idx])
            triad_v1Xs.append(v1Xs)
            
        ans_cand_idx = triad_positive_depth_counts.argmax()
        
        inlier_mask = triad_inliers_masks[ans_cand_idx] == 1
        
        w_v1Pv2 = candidates_v1Pv2[ans_cand_idx]
        v2Rv1 = candidates_v2Rv1[ans_cand_idx,:,:]
        v1Lmks = np.array(triad_v1Xs[ans_cand_idx]).squeeze()
        print("candidate idx:{:d}".format(ans_cand_idx))
        
        # update view pose
        wRv1 = self.views[idx1].R
        wPv1 = self.views[idx1].t
        self.views[idx2].R = wRv1.dot(v2Rv1.T)
        self.views[idx2].t = wPv1 + w_v1Pv2
        
        # update table (store lmk and update inlier flag)
        # if(len(self.match_manage_table) != len(inlier_mask)):
        #     raise Exception("[Error] length of match table and inlier mask is not same.")
        
        print("inlier mask len:" ,inlier_mask.shape)
        print("match manage table shape:",len(self.match_manage_table))
        
        count_triad_inliers = 0
        count_epipolar_inliers =0
        for i,dd, in enumerate(self.match_manage_table):
            
            if dd["inlier"] != InlierFlag.Inlier:
                continue

            if inlier_mask[count_epipolar_inliers] == 1:
                dd["wLmk"] = v1Lmks[count_triad_inliers]
                dd["inlier"] = InlierFlag.Inlier
                count_triad_inliers += 1
            else:
                dd["wLmk"] = None
                dd["inlier"] = InlierFlag.TriadOutlier 
                  
            count_epipolar_inliers += 1
        
        return w_v1Pv2,v2Rv1,v1Lmks,inlier_mask
        
    def bundleAdjustmentWith3views(self):
        return None