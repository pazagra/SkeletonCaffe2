import cv2
import CNNCaffe2
import time

img = cv2.imread("Test.jpg")
t1 = time.time()
CNN = CNNCaffe2.skeleton()
print "Preparation Time: "+ (time.time()-t1).__str__()+" s."
t1 = time.time()
Canvas, skeleton_dict = CNN.get_skeleton(img,[0.7])
print "Execution Time: "+ (time.time()-t1).__str__()+" s."
print "Skeleton: "
print skeleton_dict
cv2.imshow("Skeleton",Canvas)
cv2.waitKey()
