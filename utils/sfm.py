from enum import IntEnum
import copy

import cv2
from numpy.linalg import svd

import numpy as np

from feature_matcher import *
from geometric_core import *
from BA import BundleAdjust

import warnings

__IS__DEBUG__ = True

class InlierFlag(IntEnum):
    PnpOutlier = -4
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
        
        self.R_ = None
        self.t_ = None
        
        self.matching_inliers = None
        
    def set_image(self,im):
        self.im_gray = im
        
    def set_keypoint_and_descriptor(self,kp,des):
        self.key_point_uvs = kp
        self.key_point_descriptors = des    
class Sfm:
    def __init__(self,K,ims,wRv0 = np.eye(3),wPv0 = np.zeros(3),detector = "SIFT"):
        
        self.wLmks = None
        self.K = K
        self.invK = np.linalg.inv(K)
        
        self.F = None
        
        # init Bundle Adjustment 
        self.BA = BundleAdjust(K)
        
        # init views
        self.numViews = len(ims)
        
        # set view
        self.views = []
        for j,im in enumerate(ims):
            im_gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
            view = ViewBase()
            view.set_image(im_gray)
            self.views.append(view)
        
        # init view0 pose
        self.views[0].R = wRv0
        self.views[0].t = wPv0
        
        self.match_manage_table = []

        # init matcher
        self.matcher = Matcher(detector)
        
    def updateView(self,idx,view):
        self.views[idx] = view
        
    def detectKeyPoints(self,idx):
                
        kp, des = self.matcher.detector.detectAndCompute(self.views[idx].im_gray,None)
        
        self.views[idx].set_keypoint_and_descriptor(kp,des)
        
    def matchingKeyPointsOf2views(self,idx1,idx2):
        
        v1 = self.views[idx1]
        v2 = self.views[idx2]
        key_1st = "V{:d}".format(idx1)
        key_2nd = "V{:d}".format(idx2)
        
        if(v1.key_point_uvs == None or v2.key_point_uvs == None):
            raise Exception("no key points are detected.")
        
        matches = self.matcher.match(v1,v2)

        if len(self.match_manage_table) == 0:   
            print("Create New matching table")
            for i,m in enumerate(matches):
                
                data_unit = {
                            "ID":i,
                            key_1st:m.queryIdx,
                            key_2nd:m.trainIdx,
                            "wLmk":None,
                            "inlier": InlierFlag.Inlier,
                            "keypoint_match":"{:s}_{:s}".format(key_1st,key_2nd)
                        }
                
                # allocate for later views
                all_view_indexs = list(range(self.numViews))
                unproced_view_indexs = list(filter(lambda i: i !=idx1 and i != idx2, all_view_indexs))
                for k in unproced_view_indexs:
                    data_unit["V{:d}".format(k)] = None
                    
                # create new matching table
                self.match_manage_table.append(data_unit)
                    
        # append pre existed table  
        else:
             
            v1_list = [dd[key_1st] for dd in self.match_manage_table]
            
            new_points = []            
            for i,m in enumerate(matches):
                
                if m.queryIdx in v1_list:
                    dd = self.match_manage_table[v1_list.index(m.queryIdx)]
                    dd[key_2nd] = m.trainIdx
                else:
                    data_unit = {
                        "ID":i,
                        key_1st:m.queryIdx,
                        key_2nd:m.trainIdx,
                        "wLmk":None,
                        "inlier": InlierFlag.Inlier,
                        "keypoint_match":"{:s}_{:s}".format(key_1st,key_2nd)
                    }
                
                    # allocate for later views
                    all_view_indexs = list(range(self.numViews))
                    unproced_view_indexs = list(filter(lambda i: i !=idx1 or i != idx2, all_view_indexs))
                    
                    for k in unproced_view_indexs:
                        data_unit["V{:d}".format(k)] = None
                    
                    new_points.append(data_unit)
                    
            print("New {:d} points are added.".format(len(new_points)))
            self.match_manage_table.extend(new_points)
            
        return matches
                
    def get2D_2Dcoresspondance(self,idx1,idx2,isFilterInlier = False):
        
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
            
            if (self.match_manage_table[i]["keypoint_match"] == "{:s}_{:s}".format(key_1st,key_2nd)):

                pts1.append(v1.key_point_uvs[view1_array_idx].pt)
                pts2.append(v2.key_point_uvs[view2_array_idx].pt)
                GIDs.append(self.match_manage_table[i]["ID"])
                
        if len(pts1) == 0:
            warnings.warn("[warning] no matching pairs exist between view {:s} and {:s}.".format(key_1st,key_2nd))
            
        return pts1,pts2,GIDs
                
    # def get2D_2Dcoresspondance(self,idx1,idx2,isFliterInlier = False):
        
    #     v1 = self.views[idx1]
    #     v2 = self.views[idx2]
        
    #     key_1st = "V{:d}".format(idx1)
    #     key_2nd = "V{:d}".format(idx2) 

    #     pts1 = []
    #     pts2 = []
    #     GIDs = []
        
    #     n = len(self.match_manage_table)
    #     for i in range(n):
            
    #         view1_array_idx = self.match_manage_table[i][key_1st]
    #         view2_array_idx = self.match_manage_table[i][key_2nd]
            
    #         if (view1_array_idx is not None and view2_array_idx is not None):
                
    #             if isFliterInlier:
    #                 if (self.match_manage_table[i]["inlier"] == InlierFlag.Inlier):

    #                     pts1.append(v1.key_point_uvs[view1_array_idx].pt)
    #                     pts2.append(v2.key_point_uvs[view2_array_idx].pt)
    #                     GIDs.append(self.match_manage_table[i]["ID"])

    #             else:
    #                 print("ddd")
    #                 pts1.append(v1.key_point_uvs[view1_array_idx].pt)
    #                 pts2.append(v2.key_point_uvs[view2_array_idx].pt)
    #                 GIDs.append(self.match_manage_table[i]["ID"])
        
    #     self.updateView(idx1,v1)
    #     self.updateView(idx2,v2)
        
    #     return pts1,pts2,GIDs
    
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
    
    def getAllviewCoresspondance(self,isFliterInlier = False):
        
        vs = []
        keys = []
        for idx in range(self.numViews):
            vs.append(self.views[idx])
            keys.append("V{:d}".format(idx))

        views_pts = []
        lmks = []
        GIDs = []
        
        n = len(self.match_manage_table)
        for i in range(n):
            
            array_indexs = [self.match_manage_table[i][key] for key in keys]
            if all(array_indexs):
                pts = []
                if isFliterInlier:
                    if (self.match_manage_table[i]["inlier"] == InlierFlag.Inlier):
                        for v,array_index in zip(vs,array_indexs):
                            pts.append(v.key_point_uvs[array_index].pt)
                            
                        views_pts.append(pts)
                        lmks.append(self.match_manage_table[i]["wLmk"])
                        GIDs.append(self.match_manage_table[i]["ID"])

                else:
                    for v,array_index in zip(vs,array_indexs):
                        pts.append(v.key_point_uvs[array_index].pt)
                        
                    views_pts.append(pts)
                    lmks.append(self.match_manage_table[i]["wLmk"])
                    GIDs.append(self.match_manage_table[i]["ID"])
        
        lmks = np.array(lmks)
        views_pts = np.array(views_pts).transpose(1,0,2)
        
        print(lmks.shape,len(GIDs),np.array(views_pts).shape)
        
        return views_pts,lmks,GIDs      
    
    def getInliers(self,idx):
        
        view = self.views[idx]
        key = "V{:d}".format(idx)
    
    def solveEpipolarEqToGetEssentialMatrix(self,idx1,idx2):

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
                
    def solvePNPToGet_v1Tv2(self,idx1,idx2):
        
        # check wheter 1st view's pose has been solved previously.
        if self.views[idx1].R is None or self.views[idx1].t is None:
            raise Exception("[Error] 1st view's pose must be solved previously.")
        
        pts,wLmks,_ = self.get2D_3Dcoresspondance(idx1,idx2,True)
        
        if pts.shape[0] == 0 or wLmks.shape[0] == 0:
            raise Exception("size of pts and wLmks must be > 0")
        
        if pts.shape[0] != wLmks.shape[0]:
            raise Exception("shape must be same")
        
        init_R = copy.deepcopy(self.views[idx1].R)
        init_t = copy.deepcopy(self.views[idx1].t)
        print(init_t)
        wRv3,wPv3,inliers = solvePNPwRansacOutlierRejection(wLmks,pts,init_R,init_t,self.K)
        print(init_t)
        # update inlier manage table
        count = 0
        for dd in self.match_manage_table:
            if dd["inlier"] == InlierFlag.Inlier:
                count += 1
                
                if not count in inliers and dd["inlier"] == InlierFlag.EpipolarOutlier:
                    dd["inlier"] == InlierFlag.PnpOutlier
        
        # update view pose
        self.views[idx2].R = wRv3
        self.views[idx2].t = wPv3
        
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

    def landmark_propagation(self,idx_ref,idx_tgt):

        key_ref = "V{:d}".format(idx_ref)
        key_tgt = "V{:d}".format(idx_tgt)     
        v_ref = self.views[idx_ref] 
        v_tgt = self.views[idx_tgt]
        
        uvs_ref = []
        uvs_tgt = []
        for dd in self.match_manage_table:
            ref_array_idx =  dd[key_ref] 
            tgt_array_idx =  dd[key_tgt] 
            if ref_array_idx is not None and \
                tgt_array_idx is not None and \
                dd["wLmk"] is None:
                uvs_ref.append(v_ref.key_point_uvs[ref_array_idx].pt)
                uvs_tgt.append(v_tgt.key_point_uvs[tgt_array_idx].pt)
        
        if len(uvs_ref) == 0:
            print("[Warning] Matching points does not exist. No landmarks are propageted.")
            return None
        
        wRvRef = v_ref.R
        wPvRef = v_ref.t
        
        wRvTgt = v_tgt.R
        wPvTgt = v_tgt.t
        
        wLmks = []
        for i,(uvs_ref,uvs_tgt) in enumerate(zip(uvs_ref,uvs_tgt)):

            lmk = self.traiangulate_point(wRvRef,wPvRef,wRvTgt,wPvTgt,uvs_ref,uvs_tgt)
            
            if lmk[2] > 0:
                wLmks.append(lmk)
                
        wLmks = np.array(wLmks)
        
        # uvs = np.array(uvs)
        # wRc = self.views[idx].R
        # wtc = self.views[idx].t[:,np.newaxis]
        
        # uvs = np.hstack([uvs,np.ones((uvs.shape[0],1))])
        # cPxs = np.linalg.inv(self.K).dot(uvs.T)
        
        # ctw= - wRc.T.dot(wtc)
        # wPxs = wRc.T.dot((cPxs - ctw)).T
            
        return wLmks

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
        
    def make3viewGraphForBA(self):
        Rs = np.array([self.views[i].R for i in range(self.numViews)])
        Ps = np.array([self.views[i].t for i in range(self.numViews)])
        
        views_pts_inliers,wLmks,_ = self.getAllviewCoresspondance(True)
        
        # length of list check
        if len(Rs) != len(Ps) != views_pts_inliers.shape[1] != wLmks.shape[0]:
             raise Exception("The length of array and list must be the same!")
        
        # debug
        if __IS__DEBUG__:
            for wPv,wRv,pts in zip(Ps,Rs,views_pts_inliers):
                vLmks = wRv.T.dot((wLmks - wPv).T)
                XX = self.K.dot(vLmks)
                us = XX[0] / XX[2]
                vs = XX[1] / XX[2]
                
                for u,v,pt in zip(us,vs,pts):
                    print(u,v,pt)
        
        uvs = np.array(views_pts_inliers)
        self.BA.make_graph(uvs,wLmks,Rs,Ps)
        
    def optimizeBA(self,iter_num = 100):
        
        Rs,Ps = self.BA.run_optim(iter_num)
        
        for view_idx,(R,Pos) in enumerate(zip(Rs,Ps),0):
            self.views[view_idx].R_ = R
            self.views[view_idx].t_ = Pos
            
    def updatePoseToResultOfBA(self):
        
        for view_idx in range(len(self.views)):
            self.views[view_idx].R = self.views[view_idx].R_
            self.views[view_idx].t = self.views[view_idx].t_
            
    def check_reprojected_points(self,idx,use_old_buf = False,verbose=False):

        if idx == 0:
            raise Exception("idx must be > 0.")
        
        if not use_old_buf:
            wRv = self.views[idx].R
            wPv = self.views[idx].t
        else:
            wRv = self.views[idx].R_
            wPv = self.views[idx].t_
        
        pts2,wLmks,_ = self.get2D_3Dcoresspondance(idx-1,idx,True)
        pts2 = np.int32(pts2)
        canvas = copy.deepcopy(self.views[idx].im_gray)
        canvas = cv2.cvtColor(canvas,cv2.COLOR_GRAY2RGB)
        for p,v1Lmk in zip(pts2,wLmks):
            cv2.drawMarker(canvas,
                        position=p,
                        color=(0, 255, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                        line_type=cv2.LINE_4
                        )
            
            XX = self.K.dot(wRv.T.dot(v1Lmk-wPv))
            u = XX[0]/XX[2]
            v = XX[1]/XX[2]
            
            cv2.drawMarker(canvas,
                        position=(int(u),int(v)),
                        color=(255, 0, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=20,
                        thickness=2,
                        line_type=cv2.LINE_4
                        )
            if verbose:
                if 0 < u < canvas.shape[0] and 0 < v < canvas.shape[1]:
                    print((u,v),p)
                
        return canvas
