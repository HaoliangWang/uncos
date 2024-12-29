import io
import numpy as np
from PIL import Image
from os import listdir
import h5py
from os.path import isfile, join
import os
import argparse
import argparse
from PIL import Image
from uncos import UncOS
import subprocess as sp


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

# def get_gpu_memory():
#     command = "nvidia-smi --query-gpu=memory.free --format=csv"
#     memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
#     memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
#     return memory_free_values


parser = argparse.ArgumentParser(description='Download all stimuli from S3.')
parser.add_argument('--scenario', type=str, default='Dominoes', help='name of the scenarios')
args = parser.parse_args()
scenario = args.scenario
print(scenario)

# gpu_mems = get_gpu_memory()
# max_val = max(gpu_mems)
# gpu = gpu_mems.index(max_val)
# print(f"GPU: {gpu}")
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
uncos = UncOS()

source_path = '/ccn2/u/rmvenkat/data/testing_physion/regenerate_from_old_commit/test_humans_consolidated/lf_0/'
save_path = f'/ccn2/u/haw027/b3d_ipe/uncos_results/'

if scenario == 'collide':
    FINAL_T = 15
else:
    FINAL_T = 45
height = width = 350

scenario_path = join(source_path, scenario+'_all_movies')
onlyhdf5 = [f for f in listdir(scenario_path) if isfile(join(scenario_path, f)) and join(scenario_path, f).endswith('.hdf5')]
for hdf5_file in onlyhdf5:
    trial_name = '_'.join(hdf5_file.split('/')[-2:])[:-5]
    if trial_name.endswith('temp'):
        continue
    print('\t', trial_name)
    folder_name = join(save_path, scenario, trial_name)
    makedir(folder_name)

    hdf5_file_path = join(scenario_path, hdf5_file)
    with h5py.File(hdf5_file_path, "r") as f:
        # extract camera info
        camera_matrix = np.array(f['frames']['0000']['camera_matrices']['camera_matrix_cam0']).reshape((4, 4))

        vfov = 54.43222 / 180.0 * np.pi
        tan_half_vfov = np.tan(vfov / 2.0)
        tan_half_hfov = tan_half_vfov * width / float(height)
        fx = width / 2.0 / tan_half_hfov  # focal length in pixel space
        fy = height / 2.0 / tan_half_vfov

        for key in f['frames'].keys():
            if int(key)>FINAL_T:
                continue
            depth = np.array(Image.fromarray(np.array(f['frames'][key]['images']['_depth_cam0'])).resize((width, height)))
            rgb_im = np.array(Image.open(io.BytesIO(f['frames'][key]['images']['_img_cam0'][:])).resize((width, height)))
            pc = unproject_pixels(depth, camera_matrix, fx, fy)
            pc[:,[2,1]] = pc[:,[1,2]]
            pcd = pc.reshape(width, height, 3)
            try:
                pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd,
                                                                                return_most_likely_only=False,
                                                                                pointcloud_frame='world',
                                                                                n_seg_hypotheses_trial=12)
                uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses,
                                                        show=False, save_path=f'{folder_name}/{int(key)}.png')
            except Exception as e:
                print(f"!!!!!!Problematic {trial_name} frame {int(key)}")
                print(e)
                makedir(f"{save_path}/problematic/{scenario}/")
                rgb_pc = np.concatenate([rgb_im, pcd], axis=-1)
                np.save(f'{save_path}/problematic/{scenario}/{trial_name}_{int(key)}.npy', rgb_pc)
                pass