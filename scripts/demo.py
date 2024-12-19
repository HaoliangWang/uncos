import os
import glob
import argparse
from PIL import Image
import numpy as np
from uncos import UncOS
from uncos.uncos_utils import load_data_npy
import shutil
import subprocess as sp


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_files_with_extension(directory, extension):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(extension.lower()):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def main(path, vis=True, most_likely=False):
    gpu_mems = get_gpu_memory()
    max_val = max(gpu_mems)
    gpu = gpu_mems.index(max_val)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"

    test_most_likely = most_likely

    directory_path = f"/ccn2/u/haw027/b3d_ipe/uncos_vis/{path}/"
    demo_files_list = get_files_with_extension(directory_path, ".npy")
    save_path = f"/ccn2/u/haw027/b3d_ipe/uncos_results/{path}"
    
    uncos = UncOS()
    for demo_file_path in demo_files_list:
        stim_name = demo_file_path.split('/')[-1][:-4]
        frame = demo_file_path.split('/')[-2]
        makedir(os.path.join(save_path, frame))
        if f"{stim_name}_result.png" in [f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f)) and os.path.join(save_path, f).endswith('.png')]:
            continue
        try:
            rgb_im, pcd = load_data_npy(demo_file_path)
            pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd,
                                                                            return_most_likely_only=test_most_likely,
                                                                            n_seg_hypotheses_trial=12)
            if vis:
                uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)
            else:
                uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses,
                                                    show=False, save_path=f'{save_path}/{stim_name}_result.png')
                
            # for i, mask in enumerate(pred_masks_boolarray):
            #     mask_3d = np.stack((mask,mask,mask),axis=2) #3 channel mask
            #     masked_im = rgb_im*mask_3d
            #     # masked_im[mask==0] = 255 # Optional
            #     im = Image.fromarray(masked_im)
            #     im.save(f'{demo_file_path[:-4]}_masks_{i}.png')
        except Exception as e:
            print(f"!!!!!!Problematic {stim_name}")
            print(e)
            makedir(f"/ccn2/u/haw027/b3d_ipe/uncos_results/problematic/{path}/")
            shutil.move(demo_file_path, f"/ccn2/u/haw027/b3d_ipe/uncos_results/problematic/{path}/{stim_name}.npy")
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Path to npy files.')
    parser.add_argument('-v', '--vis', action='store_true', help='Visualize the segmentation result.')
    parser.add_argument('-m', '--mostlikely', action='store_true', help='Return most likely result.')
    args = parser.parse_args()
    print(f"**********************{args.path}**********************")
    main(path=args.path, vis=args.vis, most_likely=args.mostlikely)
