#!/usr/bin/env python
import os
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
import numpy as np
import cv2 as cv
import time
import struct
import yaml

def feature_detection(img,detector):
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    if detector == "Fast":
        fast = cv.FastFeatureDetector_create()
        fast.setThreshold(10)
        print(fast.getThreshold())
        kp = fast.detect(img,None)
        orb = cv.ORB_create()
        kp, des = orb.compute(img, kp)
        
    if detector == "Orb":
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
    if detector == "Sift":
        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2)
    #plt.show()
    plt.pause(0.01)
    return kp,des

def Triangulation(P1, P2, pts1, pts2):
    point_cloud = cv.triangulatePoints(P1, P2, pts1, pts2)
    point_cloud = point_cloud / point_cloud[3,:]
    return point_cloud.T

def projection_matrix(k,R,t):
    Rt = np.zeros((3,4))
    Rt[:3,:3] = R
    Rt[:,3] = t.reshape((3))
    P = k@Rt
    return P

def keypoint_matches(kp1,des1,kp2,des2,point_cloud,is_point_cloud_avaliable):
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    pts1 = cv.KeyPoint_convert(kp1)
    pts2 = cv.KeyPoint_convert(kp2)
    matches = matcher.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    src_p = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_p=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    common_des1 = np.array([ des1[m.queryIdx] for m in good ])
    common_des2 = np.array([ des2[m.trainIdx] for m in good ])
    common_pts1 = np.float32(src_p)
    common_pts2 = np.float32(dst_p)
    ckp1 = [kp1[m.queryIdx] for m in good ]
    ckp2 = [kp2[m.trainIdx] for m in good ]
    if(is_point_cloud_avaliable == True):
        print(len(point_cloud))
        print(len(good))
        point_cloud = np.array([point_cloud[m.queryIdx] for m in good ])
        return pts1,pts2,common_pts1,common_pts2,common_des1,common_des2,ckp1,ckp2,point_cloud
    else:
        return pts1,pts2,common_pts1,common_pts2,common_des1,common_des2,ckp1,ckp2

def optical_flow_matches(imgb,imga,kpa):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 5,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    kpb, st, err = cv.calcOpticalFlowPyrLK(imga, imgb, kpa, None, **lk_params)
    ptsb = kpb[st == 1]
    ptsa = kpa[st == 1]
    return ptsa.reshape(-1,1,2),ptsb.reshape(-1,1,2),st

def pnp(p3d,pts,k):
    retval, rvec, t, inliers = cv.solvePnPRansac(p3d,pts, k, (0,0,0,0),useExtrinsicGuess = True ,iterationsCount = 100,reprojectionError = 5.0,confidence = 0.750,flags = cv.SOLVEPNP_ITERATIVE)
    R,Jec = cv.Rodrigues(rvec)
    return R,t

def reprojection_error(p3d,P,pts):
    proj_pt = P@p3d.T
    proj_pt = proj_pt.T
    proj_pt= proj_pt / proj_pt[:,2].reshape((-1,1))
    proj_pt = proj_pt[:,:2]
    err = proj_pt - pts.reshape((-1,2))
    err = np.linalg.norm(err)
    err = err/len(pts)
    return err

def write_pointcloud(filename,xyz_points,rgb_points=None):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'
    # cleaning point cloud
    mean = np.mean(xyz_points[:, :3], axis=0)
    print(mean)
    temp = xyz_points[:, :3] - mean
    print(temp.mean(axis = 0))
    dist = np.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2 + temp[:, 2] ** 2)
    #print(dist.shape, np.mean(dist))
    indx = np.where(dist < np.mean(dist)+ 2)
    xyz_points = xyz_points[indx]
    rgb_points = rgb_points[indx]
    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,2].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,0].tobytes())))
    fid.close()

def VO(L):
    start = time.time()
    print("no.of input frames")
    print(len(L))
    point_cloud = []
    trajectory = []
    is_point_cloud_avaliable = False
    color = cv.imread(dataset_location +'/'+ L[0])
    plt.imshow(color)
    plt.show()
    img1 = cv.imread(dataset_location +'/'+ L[0],cv.IMREAD_GRAYSCALE)
    kp1,des1 = feature_detection(img1,"Sift")
    img2 = cv.imread(dataset_location +'/'+ L[1],cv.IMREAD_GRAYSCALE)
    kp2,des2 = feature_detection(img2,"Sift")
    pts1,pts2,common_pts1,common_pts2,common_des1,common_des2,ckp1,ckp2 = keypoint_matches(kp1,des1,kp2,des2,point_cloud,is_point_cloud_avaliable) 
    E,mask = cv.findEssentialMat(common_pts1,common_pts2,k,cv.RANSAC,0.999,1.0,1000)
    common_pts1 = common_pts1[mask.ravel() == 1]
    common_pts2 = common_pts2[mask.ravel() == 1]
    common_des1 = common_des1[mask.ravel() == 1]
    common_des2 = common_des2[mask.ravel() == 1]
    retval, R, t, mask = cv.recoverPose(E, common_pts1, common_pts2, k)
    common_pts1 = common_pts1[mask.ravel() == 255]
    common_pts2 = common_pts2[mask.ravel() == 255]
    common_des1 = common_des1[mask.ravel() == 255]
    common_des2 = common_des2[mask.ravel() == 255]
    ckp1 = cv.KeyPoint_convert(common_pts1)
    ckp2 = cv.KeyPoint_convert(common_pts2)
    print(len(common_pts1),len(common_pts2))
    P1 = projection_matrix(k,np.eye(3).astype(np.float32),np.zeros((3,1))) 
    P2 = projection_matrix(k,R,t) 
    trajectory.append(t)
    print("trigulating img 1 and 2")
    point_cloud = Triangulation(P1, P2, common_pts1, common_pts2)
    print(0,reprojection_error(point_cloud,P1,common_pts1))
    print(1,reprojection_error(point_cloud,P2,common_pts2))
    p3d = point_cloud
    idx = common_pts1.reshape(-1,2).astype(np.uint8())
    color_data = color[idx[:,0],idx[:,1]]
    print(color)
    print(p3d[:,:3].shape)
    #write_pointcloud("abc.ply",p3d[:,:3],color[idx[:,0],idx[:,1]])
    #print(point_cloud.shape)
    for i in range(1,2):
        print(i)
        is_point_cloud_avaliable = True
        print(L[i],L[i+1])
        color = cv.imread(dataset_location +'/'+ L[i])
        img3 = cv.imread(dataset_location +'/'+ L[i+1],cv.IMREAD_GRAYSCALE)
        kp3,des3 = feature_detection(img2,"Sift")
        #print(len(ckp2))
        pts2,pts3,common_pts2,common_pts3,common_des2,common_des3,ckp2,ckp3,point_cloud = keypoint_matches(ckp2,common_des2,kp3,des3,point_cloud,is_point_cloud_avaliable)
        print(point_cloud.shape)
        R1,t1 = pnp(point_cloud[:,:3],common_pts3,k)
        is_point_cloud_avaliable = False
        print(R,t,R1,t1)
        trajectory.append(t1)
        P3 = projection_matrix(k,R1,t1)
        #if(np.linalg.norm(t - t1)> 0.01):
        print('not a same')
        pts2,pts3,common_pts2,common_pts3,common_des2,common_des3,ckp2,ckp3 = keypoint_matches(kp2,des2,kp3,des3,point_cloud,is_point_cloud_avaliable)
        point_cloud = Triangulation(P2, P3, common_pts2, common_pts3)
        print(i,reprojection_error(point_cloud,P3,common_pts3))
        idx = common_pts2.reshape(-1,2).astype(np.uint8())
        if(reprojection_error(point_cloud,P3,common_pts3) < 5):
            p3d = np.vstack((p3d,point_cloud))
            print(color[idx[:,0],idx[:,1]])
            color_data = np.vstack((color_data,color[idx[:,0],idx[:,1]]))
        img2 = img3
        kp2,des2 = kp3,des3
        pts2,common_pts2,common_des2,ckp2 = pts3,common_pts3,common_des3,ckp3
        #print(len(ckp2))
        #print(len(point_cloud))
        P2 = P3
    end = time.time()
    print ("Time elapsed:", end - start)
    
    write_pointcloud("pt.ply",p3d[:,:3],color_data)

def find_trajectory(path,L):
    start = time.time()
    point_cloud = []
    color_data = []
    trajectory = []
    is_point_cloud_avaliable = False

    frame = 0
    img_color = cv.imread(path +'/'+ L[frame])
    print(frame)
    img0 = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    frame = frame + 1
    kp0,des0 = feature_detection(img0,"Sift")
    print(frame)
    img1 = cv.imread(path +'/'+ L[frame],0)
    frame = frame + 1
    kp1,des1 = feature_detection(img1,"Sift")

    pts0,pts1,common_pts0,common_pts1,common_des0,common_des1,ckp0,ckp1 = keypoint_matches(kp0,des0,kp1,des1,point_cloud,is_point_cloud_avaliable) 
    
    E,mask = cv.findEssentialMat(common_pts0,common_pts1,k,cv.RANSAC,0.999,1.0,1000)
    common_pts0 = common_pts0[mask.ravel() == 1]
    common_pts1 = common_pts1[mask.ravel() == 1]
    common_des0 = common_des0[mask.ravel() == 1]
    common_des1 = common_des1[mask.ravel() == 1]
    
    retval, R, t, mask = cv.recoverPose(E, common_pts0, common_pts1, k)
    common_pts0 = common_pts0[mask.ravel() == 255]
    common_pts1 = common_pts1[mask.ravel() == 255]
    common_des0 = common_des0[mask.ravel() == 255]
    common_des1 = common_des1[mask.ravel() == 255]
    ckp0 = cv.KeyPoint_convert(common_pts0)
    ckp1 = cv.KeyPoint_convert(common_pts1)
    
    print(len(common_pts0),len(common_pts1))
    P1 = projection_matrix(k,np.eye(3).astype(np.float32),np.zeros((3,1))) 
    P2 = projection_matrix(k,R,t)
    point_cloud = Triangulation(P1, P2, common_pts0, common_pts1)
    
    print(0,reprojection_error(point_cloud,P1,common_pts0))
    print(1,reprojection_error(point_cloud,P2,common_pts1))
    
    p3d = point_cloud
    idx = common_pts0.reshape(-1,2).astype(np.uint8())
    color_data = img_color[idx[:,0],idx[:,1]]
    #print(color_data)
    while(frame<len(L)):
        print(frame)
        is_point_cloud_avaliable = True
        img1_color = cv.imread(path +'/'+ L[frame - 1])
        img2 = cv.imread(path +'/'+ L[frame],cv.IMREAD_GRAYSCALE)
        kp2,des2 = feature_detection(img2,"Sift")

        pts1,pts2,common_pts1,common_pts2,common_des1,common_des2,ckp1,ckp2,point_cloud = keypoint_matches(ckp1,common_des1,kp2,des2,point_cloud,is_point_cloud_avaliable)

        R1,t1 = pnp(point_cloud[:,:3],common_pts2,k)
        is_point_cloud_avaliable = False

        trajectory.append(t1)

        P3 = projection_matrix(k,R1,t1)

        pts1,pts2,common_pts1,common_pts2,common_des1,common_des2,ckp1,ckp2 = keypoint_matches(kp1,des1,kp2,des2,point_cloud,is_point_cloud_avaliable)
        point_cloud = Triangulation(P2, P3, common_pts1, common_pts2)

        print(frame,reprojection_error(point_cloud,P3,common_pts2))

        idx = common_pts1.reshape(-1,2).astype(np.uint8())

        p3d = np.vstack((p3d,point_cloud))

        color_data = np.vstack((color_data,img1_color[idx[:,0],idx[:,1]]))
        img1 = img2
        kp1,des1 = kp2,des2
        pts1,common_pts1,common_des1,ckp1 = pts2,common_pts2,common_des2,ckp2
        #print(len(ckp2))
        #print(len(point_cloud))
        P2 = P3
        frame = frame + 1
    write_pointcloud("output.ply",p3d[:,:3],color_data.reshape(-1,3))
    traj = np.asarray(trajectory).reshape((-1,3))
    plt.figure()
    plt.subplot(1, 1, 1, projection='3d').plot(traj[:, 0], traj[:, 1], traj[:, 2])
    plt.show()


if __name__ == "__main__":
  with open('info.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
  path = data["dataset_location"]
  k = np.array(data["k"])
  L = os.listdir(path)
  L.sort()
  find_trajectory(path,L)
  
