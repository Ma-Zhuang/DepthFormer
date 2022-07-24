import sys
import numpy as np
from IPython import embed
import cv2

def gen_rgb(value, minimum=0, maximum=255):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 8 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255-b-r
    return r,g,b

def get_calib(img_path,gt_path,calib_path):
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path,-1)/256
    with open(calib_path,'r') as f:
        calib = f.readlines()
    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    return img, gt, P2, R0_rect, Tr_velo_to_cam

def gen_pe(img, P2, R0_rect, Tr_velo_to_cam):
    pe_depth = np.ones((img.shape[0], img.shape[1]))*200
    A = P2 * R0_rect * Tr_velo_to_cam
    R_inv = np.linalg.inv(A[0:3,0:3])
    T = A[0:3,3]
    RT =  R_inv * T
    for i in range(img.shape[1]):
        for j in range(img.shape[0]):
            t = RT[2]-1.73/(R_inv[2,0]*i+R_inv[2,1]*j+R_inv[2,2])
            if t < 0:continue
            pe_depth[j][i] = t
    return pe_depth

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    calib_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/cam_to_world.txt"
    img_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000007.png"
    gt_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/gt_depth/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000007.png"
    
    img, gt, P2, R0_rect, Tr_velo_to_cam = get_calib(img_path,gt_path,calib_path)
    pe_depth = gen_pe(img, P2, R0_rect, Tr_velo_to_cam)
    # abs_list = []
    # for i in range(img.shape[1]):
    #     for j in range(img.shape[0]):
    #         # if not gt[j,i] == 0:
    #         #     print(gt[j,i],' | ',pe_depth[j,i])
    #         #     err = (abs(pe_depth[j,i]-gt[j,i]))/(gt[j,i])
    #         #     abs_list.append(round(err,2))
    #         #     if err <=0.05:
    #         #         cv2.circle(img, (i,j), 1, gen_rgb(pe_depth[j,i]), -1)
    #         print(gt[j,i],' | ',pe_depth[j,i])
    #         err = (abs(pe_depth[j,i]-gt[j,i]))/(gt[j,i])
    #         abs_list.append(round(err,2))
    #         if err <=0.05:
    #             cv2.circle(img, (i,j), 1, gen_rgb(pe_depth[j,i]), -1)
    # cv2.imwrite('./kitti_pe_vis_new.png',img)
    pe_depth = (pe_depth*100).astype(np.uint16)
    cv2.imwrite('/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/pe.png',pe_depth)
    # num_dict = dict()
    # for err in np.unique(abs_list):
    #     num_dict[str(err)] = abs_list.count(err)
    
    embed()
    exit()

    # num_name_dict = []
    # num_list = []
    # for err in np.unique(abs_list):
    #     # num_dict[str(err)] = abs_list.count(err)
    #     num_list.append(abs_list.count(err))
    #     num_name_dict.append(str(err))
    
    # embed()
    # exit()
    # plt.xlabel('ERROR')
    # plt.ylabel('NUM')

    # first_bar = plt.bar(range(len(num_list)), num_list, color='blue')

    # plt.xticks(index, name_list)
    # plt.savefig('./pe_num.jpg')


    