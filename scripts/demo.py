import os
import glob
import argparse
from PIL import Image
import numpy as np
from uncos import UncOS
from uncos.uncos_utils import load_data_npy
import shutil


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(path, vis=True, most_likely=False):
    test_most_likely = most_likely

    demo_files_list = glob.glob(os.path.join(os.path.dirname(__file__), f"../haoliang/{path}/**.npy"))
    save_path = f"../haoliang/{path}_results"
    makedir(save_path)
    uncos = UncOS()
    for demo_file_path in demo_files_list:
        stim_name = demo_file_path.split('/')[-1][:-4]
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
            shutil.move(demo_file_path, f"../haoliang/problematic/{stim_name}_npy")
            pass

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--path', help='Path to npy files.')
    # parser.add_argument('-v', '--vis', action='store_true', help='Visualize the segmentation result.')
    # parser.add_argument('-m', '--mostlikely', action='store_true', help='Return most likely result.')
    # args = parser.parse_args()
    for path in ["collide", "contain", "drop", "dominoes", "link", "roll", "support"]:
        print(f"**********************{path}**********************")
        main(path=path, vis=False, most_likely=False)
    # main(path=args.path, vis=args.vis, most_likely=args.mostlikely)
