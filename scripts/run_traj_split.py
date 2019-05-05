"""
Launch from the project folder.
Choices of k
    -3: two randomly sampled MNIST frames.
    -2: one rope one noise.
    -1: two randomly sampled rope frames.
    0: two identical rope frames, 0 step apart
    n: two rope frames are n steps apart.
Examples:
DCGAN
    python doodad_causal_infogan_rope_continuous.py -dc 0 -rn 100 -infow 0 -gtype 60 -prefix dcgan
"""

import os
import sys

import random
import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
import colored_traceback.always
from doodad.utils import REPO_DIR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-ec2", action="store_true")
parser.add_argument("-savepath", help="output path for local run.", type=str, default='/home/thanard/rope/causal_infogan_continuous')
parser.add_argument("-loadpath", help="load parameters from path.", type=str, default='', nargs="+")
parser.add_argument("-loaditer", type=int, help="Use with loadpath", nargs="+")
parser.add_argument("-fcnpath", help="load fcn parameters from path", type=str,
                    default='/home/thanard/Downloads/FCN_mse')
parser.add_argument("-target", type=str, default='causal_infogan_rope_continuous/main.py')
parser.add_argument("-data_dir", type=str,
                    default='/home/thanard/Downloads/rope_full')
parser.add_argument("-planning_data_dir", type=str,
                    default='/home/thanard/Downloads/seq_data_2')
                    # default='/home/thanard/causal-infogan-rope/240x240/train')
parser.add_argument("-prefix", type=str, default=None)
parser.add_argument("-seed", type=int, default=None)
parser.add_argument("-n", type=int, default=3)

# Training Hyperparameters
parser.add_argument("-niters", type=int, default=100)
parser.add_argument("-dc", type=int, default=0)
parser.add_argument("-cc", type=int, default=7)
parser.add_argument("-rn", type=int, default=4,
                    help="dimension of random noise")
parser.add_argument("-infow", type=float, default=0.1, help="weights of information term.")
parser.add_argument("-transw", type=float, default=0.1, help="weights of transition regularization term.")
parser.add_argument("-vaew", type=float, default=0.0, help="weights of reconstruction term.")
parser.add_argument("-auxw", type=float, default=0.0, help="weights of D(real, fake) term.")
parser.add_argument("-lr_d", type=float, default=0.0002)
parser.add_argument("-lr_g", type=float, default=0.0002)
parser.add_argument("-gtype", type=int, default=6,
                    help="which type of generator to choose from: 6 or 70")
parser.add_argument("-dtype", type=int, default=3,
                    help="which type of discriminator to choose from: 0 to 4")
parser.add_argument("-tsize", type=int, default=[64, 64], nargs="+",
                    help="hidden size of Transition NN.")
parser.add_argument("-k", type=int, default=1,
                    help="which dataset configurations to choose from: -3 to +inf.")
parser.add_argument("-color", action="store_true")
parser.add_argument("-red", action="store_true")
parser.add_argument("-pretrain", type=str, default=None, help="pretrain the generators.")
parser.add_argument("-a_dim", type=int, default=3)
parser.add_argument("-q_eval", action="store_true")
parser.add_argument("-learn_mu", action="store_true")
parser.add_argument("-learn_var", action="store_true")

# Planning
parser.add_argument("-planning_iter", type=int, default=[100], nargs="+",
                    help="List of iterations we plan.")
parser.add_argument("-planning_horizon", type=int, default=10,
                    help="Set to 0 if doesn't run planning.")
parser.add_argument("-traj_eval_copies", type=int, default=100,
                    help='the number of plans to choose from.')
parser.add_argument("-discretization_bins", type=int, default=20)
parser.add_argument("-planner", type=str, default='simple_plan',
                    help="either simple_plan or astar_plan")
parser.add_argument("-load_D", type=str,
                    default="/home/thanard/Downloads/k1-km1-tkm1-kcoNone-dp-None-l2-1e-05-sd2000/modules/0012_Conv2d.pkl",
                    help="Help selecting k best plans.")

args = parser.parse_args()

if args.prefix is None:
    str_list = ["continuous",
                "gtype", str(args.gtype),
                "rn", str(args.rn),
                "cc", str(args.cc),
                "infow", "%.2f" % args.infow,
                "transw", "%.2f" % args.transw,
                "vaew", "%.2f" % args.vaew,
                "auxw", "%.2f" % args.auxw,
                ]
    if args.planning_horizon > 0 and os.path.exists(args.planning_data_dir) and args.planner:
        str_list.append(args.planner)
    if args.fcnpath:
        str_list.append("fcn")
    if args.q_eval:
        str_list.append("q-eval")
    if args.learn_mu:
        str_list.append("mu")
    if args.learn_var:
        str_list.append("var")
    if args.pretrain:
        str_list.append("pretrain")
        str_list.append("infogan")
    args.prefix = "-".join(str_list)
    print("Experiment name : ", args.prefix)


# project_dir = os.path.abspath(os.path.join(args.target, os.pardir))
project_dir = os.path.abspath('.')
excluded_dir = os.listdir(project_dir)
excluded_dir.remove('causal_infogan_rope_continuous')

kwargs={'n_epochs': args.niters,
        'disc_code_dim': args.dc,
        'cont_code_dim': args.cc,
        'random_noise_dim': args.rn,
        'infow': args.infow,
        'transw': args.transw,
        'vaew': args.vaew,
        'auxw': args.auxw,
        'lr_d': args.lr_d,
        'lr_g': args.lr_g,
        'gtype': args.gtype,
        'dtype': args.dtype,
        'k': args.k,
        'gray': not args.color,
        'tsize': args.tsize,
        'red': args.red,
        'pretrain': args.pretrain,
        'loadpath': args.loadpath,
        'loaditer': args.loaditer,
        'a_dim': args.a_dim,
        'fcnpath': args.fcnpath,
        'q_eval': args.q_eval,
        'learn_mu': args.learn_mu,
        'learn_var': args.learn_var,
        'planning_iter': args.planning_iter,
        'discretization_bins': args.discretization_bins,
        'traj_eval_copies': args.traj_eval_copies,
        'planning_horizon': args.planning_horizon,
        'planner': args.planner,
        'python_cmd': " ".join(sys.argv),
        'load_D': args.load_D
        }

# import json
# with open('causal_infogan_rope_continuous/params.json', 'r') as f:
#     tmp = json.load(f)
# kwargs.update(tmp)

if args.ec2:
    # Appending prefix name.
    with open("/home/thanard/rope/ec2/running_experiments.txt", "a") as myfile:
        myfile.write(args.prefix + "\n")

    MY_RUN_MODE = dd.mode.EC2AutoconfigDocker(
        image='thanard/matplotlib:latest',
        region='us-east-1',
        # zone='us-east-1b',
        instance_type='p2.xlarge',
        spot_price=1.0,
        s3_log_prefix=args.prefix,
        gpu=True,
        terminate=True,
        pre_cmd=['pip install tensorboard_logger',
                 'pip install tensorflow',
                 'pip install networkx',
                 'pip install astar',
                 'pip install dill',
                 'pip install opencv-python',
                 'apt update && apt install -y libsm6 libxext6 libxrender1'],
    )

    # Set up code and output directories
    OUTPUT_DIR = '/example/outputs'  # this is the directory visible to the target in docker
    # DATA_DIR = '/home/ubuntu/rope_full' # this is the directory visible to the target in docker
    ec2_data_dir = '/home/ubuntu/rope_full'
    DATA_DIR = args.data_dir # this is the directory visible to the target in docker
    # Because imgs_skipped contains local data_dir paths.
    PLANNING_DATA_DIR = args.planning_data_dir
    mounts = [
        mount.MountLocal(local_dir=REPO_DIR, mount_point='/root/code/doodad', pythonpath=True),
        # mount.MountLocal(local_dir='/home/thanard/Downloads/rllab/sandbox/thanard/infoGAN', pythonpath=True),
        mount.MountLocal(local_dir=project_dir, filter_dir=excluded_dir, pythonpath=True),
        mount.MountEC2(ec2_path=ec2_data_dir, mount_point=DATA_DIR),
        mount.MountS3(s3_path='', mount_point=OUTPUT_DIR, output=True, include_types=('*.png', '*.log', '*.txt', '*.csv', '*.json')),
    ]
    local_img_tmp = os.path.join(args.data_dir, 'imgs_skipped_%d.pkl' % args.k)
    if os.path.exists(local_img_tmp):
        mounts.append(mount.MountLocal(
            local_dir=local_img_tmp,
            mount_point=os.path.join(DATA_DIR, 'imgs_skipped_%d.pkl' % args.k)))
    if kwargs['loadpath']:
        for name in ['G', 'D', 'Gaussian_Posterior', 'Gaussian_Transition']:
            for path, iter in zip(kwargs['loadpath'], kwargs['loaditer']):
                mounts.append(mount.MountLocal(local_dir=os.path.join(path,
                                            'var/%s_%d' % (name, iter)),
                                             mount_point=os.path.join(path,
                                             'var/%s_%d' % (name, iter))
                                             ))
    if kwargs['load_D'] and kwargs['planning_horizon'] > 0:
        mounts.append(mount.MountLocal(local_dir=kwargs['load_D'],
                                       mount_point=kwargs['load_D']))
    if os.path.exists(args.planning_data_dir):
        mounts.append(mount.MountLocal(local_dir=args.planning_data_dir,
                                       mount_point=PLANNING_DATA_DIR))
    if kwargs['fcnpath']:
        mounts.append(mount.MountLocal(local_dir=args.fcnpath, mount_point=args.fcnpath))
    print(mounts)

    THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))
    for i in range(args.n):
        dd.launch_python(
            target=os.path.abspath(args.target),
            mode=MY_RUN_MODE,
            mount_points=mounts,
            args={

                'data_dir': DATA_DIR,
                'output_dir': OUTPUT_DIR,
                'seed': i,
                'kwargs': kwargs,
                'ec2': True,
                'planning_data_dir': PLANNING_DATA_DIR
            },
            dry=False,
            verbose=True,
            postfix='%03d' % i,
            # postfix only works with my modified doodad.
        )
else:
    mounts = [
        mount.MountLocal(local_dir=project_dir, pythonpath=True),
    ]
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    dd.launch_python(
        target=os.path.abspath(args.target),
        mount_points=mounts,
        args={
            'data_dir': args.data_dir,
            'output_dir': os.path.join(args.savepath, args.prefix),
            'seed': args.seed,
            'kwargs': kwargs,
            'ec2': False,
            'planning_data_dir': args.planning_data_dir
        }
    )
