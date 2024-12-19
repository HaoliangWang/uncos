import io
import numpy as np
from PIL import Image
from os import listdir
import h5py
from os.path import isfile, join
import os
import argparse


def unproject_pixels(depth, cam_matrix, fx, fy):
    '''
    pts: [N, 2] pixel coords
    depth: [N, ] depth values
    returns: [N, 3] world coords
    '''
    mask = np.ones(depth.shape[:2], dtype=bool)
    pts = np.array([[x,y] for x,y in zip(np.nonzero(mask)[0], np.nonzero(mask)[1])])
    camera_matrix = np.linalg.inv(cam_matrix.reshape((4, 4)))

    # Different from real-world camera coordinate system.
    # OpenGL uses negative z axis as the camera front direction.
    # x axes are same, hence y axis is reversed as well.
    # Source: https://learnopengl.com/Getting-started/Camera
    rot = np.array([[1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1]])
    camera_matrix = np.dot(camera_matrix, rot)


    height = depth.shape[0]
    width = depth.shape[1]

    img_pixs = pts[:, [1, 0]].T
    img_pix_ones = np.concatenate((img_pixs, np.ones((1, img_pixs.shape[1]))))

    # Calculate the intrinsic matrix from vertical_fov.
    # Motice that hfov and vfov are different if height != width
    # We can also get the intrinsic matrix from opengl's perspective matrix.
    # http://kgeorge.github.io/2014/03/08/calculating-opengl-perspective-matrix-from-opencv-intrinsic-matrix
    intrinsics = np.array([[fx, 0, width/ 2.0],
                           [0, fy, height / 2.0],
                           [0, 0, 1]])
    img_inv = np.linalg.inv(intrinsics[:3, :3])
    cam_img_mat = np.dot(img_inv, img_pix_ones)

    points_in_cam = np.multiply(cam_img_mat, depth.reshape(-1))
    points_in_cam = np.concatenate((points_in_cam, np.ones((1, points_in_cam.shape[1]))), axis=0)
    points_in_world = np.dot(camera_matrix, points_in_cam)
    points_in_world = points_in_world[:3, :].T#.reshape(3, height, width)
    
    return points_in_world

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser(description='Download all stimuli from S3.')
parser.add_argument('--scenario', type=str, default='Dominoes', help='name of the scenarios')
args = parser.parse_args()
scenario = args.scenario
print(scenario)


source_path = '/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0/'
save_path = '/ccn2/u/haw027/b3d_ipe/uncos_vis/'

if scenario == 'collide':
    FINAL_T = 15
else:
    FINAL_T = 45
w = h = 350

scenario_path = join(source_path, scenario+'_all_movies')
onlyhdf5 = [f for f in listdir(scenario_path) if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith('.hdf5')]
for hdf5_file in onlyhdf5:
    trial_name = '_'.join(hdf5_file.split('/')[-2:])[:-5]
    if trial_name.endswith('temp'):
        continue
    print('\t', trial_name)

    hdf5_file_path = join(scenario_path, hdf5_file)
    depth_arr = []
    image_arr = []
    with h5py.File(hdf5_file_path, "r") as f:
        for key in f['frames'].keys():
            if int(key)>FINAL_T:
                continue
            depth = np.array(Image.fromarray(np.array(f['frames'][key]['images']['_depth_cam0'])).resize((w, h)))
            depth_arr.append(depth)
            image = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_img_cam0'][:])).resize((w, h)))
            image_arr.append(image)
        depth_arr = np.asarray(depth_arr)
        image_arr = np.asarray(image_arr)
        height, width = image_arr.shape[1], image_arr.shape[2]

        # # extract depth info
        # depth = np.array(f['frames']['0000']['images']['_depth_cam0'])
        # image = np.array(Image.open(io.BytesIO(f['frames']['0000']['images']['_img_cam0'][:])))
        # height, width = image.shape[0], image.shape[1]

        # extract camera info
        camera_matrix = np.array(f['frames']['0000']['camera_matrices']['camera_matrix_cam0']).reshape((4, 4))

        vfov = 54.43222 / 180.0 * np.pi
        tan_half_vfov = np.tan(vfov / 2.0)
        tan_half_hfov = tan_half_vfov * width / float(height)
        fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
        fy = height / 2.0 / tan_half_vfov

    for i, (depth, image) in enumerate(zip(depth_arr, image_arr)):
        folder_name = join(save_path, scenario, str(i))
        makedir(folder_name)
        pc = unproject_pixels(depth, camera_matrix, fx, fy)
        pc[:,[2,1]] = pc[:,[1,2]]
        non_flat = pc.reshape(width, height, 3)
        rgb_pc = np.concatenate([image, non_flat], axis=-1)
        np.save(f'{folder_name}/{trial_name}.npy', rgb_pc)
        # np.save(f'/home/haoliangwang/uncos/haoliang/test/{trial_name}.npy', rgb_pc)