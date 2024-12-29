import os
import glob
import argparse
from PIL import Image
import numpy as np
from uncos import UncOS
from uncos.uncos_utils import load_data_npy


def main(scenario, stim_name, vis=True, most_likely=False):
    test_most_likely = most_likely

    demo_file_path = f"/home/haw027/code/uncos/scripts/pilot-containment-torus_0038.npy"
    save_path = f"/home/haw027/code/uncos/scripts/"
    
    uncos = UncOS()
    rgb_im, pcd = load_data_npy(demo_file_path)
    pred_masks_boolarray, uncertain_hypotheses = uncos.segment_scene(rgb_im, pcd,
                                                                    return_most_likely_only=test_most_likely,
                                                                    pointcloud_frame='world',
                                                                    n_seg_hypotheses_trial=12)
    if vis:
        uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses)
    else:
        uncos.visualize_confident_uncertain(pred_masks_boolarray, uncertain_hypotheses,
                                            show=False, save_path=f'{save_path}/{stim_name}.png')
        
    # for i, mask in enumerate(pred_masks_boolarray):
    #     mask_3d = np.stack((mask,mask,mask),axis=2) #3 channel mask
    #     masked_im = rgb_im*mask_3d
    #     # masked_im[mask==0] = 255 # Optional
    #     im = Image.fromarray(masked_im)
    #     im.save(f'{demo_file_path[:-4]}_masks_{i}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', help='Path to npy files.')
    parser.add_argument('-t', '--stim_name', help='Path to npy files.')
    parser.add_argument('-v', '--vis', action='store_true', help='Visualize the segmentation result.')
    parser.add_argument('-m', '--mostlikely', action='store_true', help='Return most likely result.')
    args = parser.parse_args()
    print(f"**********************{args.stim_name}**********************")
    main(scenario=args.scenario, stim_name=args.stim_name, vis=args.vis, most_likely=args.mostlikely)
