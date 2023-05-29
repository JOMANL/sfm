import cv2
import numpy as np

def solvePNPwRansacOutlierRejection(lmks,uvs,init_R,init_t,K,dist_coeffs= np.zeros(shape=[8, 1], dtype='float64')):
    
    init_rot_vec,_ = cv2.Rodrigues(init_R)
    
    (success, rot_vec, trans_vec,inliers) = cv2.solvePnPRansac(lmks, uvs.astype(np.float64), K.astype(np.double), 
                                                        dist_coeffs,
                                                        reprojectionError=5,
                                                        iterationsCount=10000,
                                                        rvec=init_rot_vec,tvec=init_t,
                                                        flags=cv2.SOLVEPNP_EPNP,useExtrinsicGuess = True)

    inliers = np.array(inliers).squeeze()
    
    uvs_inlier = np.take(uvs,inliers,axis=0)
    lmks_inlier = np.take(lmks,inliers,axis=0)
    
    (success, rot_vec, trans_vec) = cv2.solvePnP(
        lmks_inlier.astype(np.float64), uvs_inlier.astype(np.float64), K.astype(np.double), 
                                                            dist_coeffs,
                                                            rot_vec,trans_vec,
                                                            flags=cv2.SOLVEPNP_ITERATIVE,useExtrinsicGuess = True)
        
    if not success:
        raise Exception("[Error] Fail to PNP")
    
    ret_R = cv2.Rodrigues(rot_vec)[0].T
    ret_t = -ret_R.dot(trans_vec.T)
    
    return ret_R,ret_t,inliers 