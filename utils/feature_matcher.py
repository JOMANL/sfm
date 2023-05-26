import cv2

class Matcher:
    def __init__(self,detector = "SIFT"):
        
        self.detector_name = detector
        
        if detector == "SIFT":
            self.detector = cv2.SIFT_create()
            self.kp_matcher = cv2.BFMatcher()
        elif detector == "ORB":
            self.detector = cv2.ORB_create()
            self.kp_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif detector == "AKAZE":
            self.detector = cv2.AKAZE_create()
            self.kp_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        else:
            raise Exception("unspported detector {:s}!".format(detector))
        
    def match(self,v1,v2,th_ratio = 0.5):
        
        if self.detector_name == "SIFT":
            matches_ = self.kp_matcher.knnMatch(v1.key_point_descriptors,v2.key_point_descriptors, k=2)
            matches = []
            
            for match1,match2 in matches_:
                if match1.distance < th_ratio*match2.distance:
                    matches.append(match1)
                    
        elif self.detector_name == "ORB":
            matches = self.kp_matcher.match(v1.key_point_descriptors,v2.key_point_descriptors)
            
        elif self.detector_name == "AKAZE":
            matches = self.kp_matcher.match(v1.key_point_descriptors,v2.key_point_descriptors)

        return matches
        