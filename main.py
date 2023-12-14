import subprocess
import time
import argparse
'''
parser = argparse.ArgumentParser(description='Measurements')
parser.add_argument('-ht', '--height', type=str, default=170.5, help='Custome height of body')

args=parser.parse_args()
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Full inference script")

    parser.add_argument(
        "--category",
        type=str,
        default="dress",
        help="cloth category for virtual try on",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="human image path for virtual try on",
    )

    parser.add_argument(
        "--cloth",
        type=str,
        default="",
        help="cloth image path for virtual try on",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

import os

def run_script0(script_path, conda_env, category, image, cloth):
    # Activate the Conda environment
    activate_cmd = f'conda activate {conda_env} && '
    
    # Execute the script within the environment
    subprocess.run(activate_cmd + 'python ' + script_path + ' --category ' + category + ' --image ' + image + ' --cloth ' + cloth, shell=True, check=True)

def run_script1(script_path, conda_env):
    # Activate the Conda environment
    activate_cmd = f'conda activate {conda_env} && '
    
    # Execute the script within the environment
    subprocess.run(activate_cmd + 'python ' + script_path, shell=True, check=True)

def run_script2(script_path, conda_env, category):
    # Activate the Conda environment
    activate_cmd = f'conda activate {conda_env} && '
    
    # Execute the script within the environment
    subprocess.run(activate_cmd + 'python ' + script_path + ' show ' + './preprocess/detectron2/projects/DensePose/configs/densepose_rcnn_R_50_FPN_s1x.yaml ' + './preprocess/detectron2/projects/DensePose/model_final_162be9.pkl ' + './input/' + category + '/images dp_segm ' + '-v --output ' + './input/' + category + '/dense/', shell=True, check=True)

def run_script3(script_path, conda_env, category):
    # Activate the Conda environment
    activate_cmd = f'conda activate {conda_env} && '
    
    # Execute the script within the environment
    subprocess.run(activate_cmd + 'python ' + script_path + ' --category ' + category, shell=True, check=True)

# Paths to your Python scripts
script1_path = './img_preprocess.py'
script2_path = './preprocess/detectron2/projects/DensePose/apply_net.py'
script3_path = './preprocess/pytorch-openpose/process.py'
script4_path = './preprocess/Self-Correction-Human-Parsing/simple_extractor.py'
script5_path = './dense_preprocess.py'
script6_path = './load_cloth.py'
script7_path = './cloth_mask.py'
script8_path = './src/inference.py'
script9_path = './Refinement.py'

# Names of the Conda environments
env1_conda_env = 'base'
env2_conda_env = 'densepose-vto'
#env3_conda_env = 'pytorch-openpose'
env3_conda_env = 'openpose'
env4_conda_env = 'label_maps_vto'

env5_conda_env = 'base'
env6_conda_env = 'base'
env7_conda_env = 'base'
env8_conda_env = 'full-vto'
env9_conda_env = 'full-vto'

start_time = time.time()

args = parse_args()

category = args.category

run_script0(script1_path, env1_conda_env, args.category, args.image, args.cloth)
run_script2(script2_path, env2_conda_env, category)
run_script1(script3_path, env3_conda_env)
run_script3(script4_path, env4_conda_env, category)
run_script1(script5_path, env5_conda_env)
run_script1(script6_path, env6_conda_env)
run_script1(script7_path, env7_conda_env)
run_script3(script8_path, env8_conda_env, category)
run_script3(script9_path, env9_conda_env, category)

end_time = time.time()  # End time for measuring program execution
total_time = end_time - start_time
print(f"Total time taken: {total_time} seconds")
