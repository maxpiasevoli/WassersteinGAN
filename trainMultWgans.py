# As of now, this file only generates WGANs with the convolutional architecture
# but it can easily be changed to automate training of mlp WGANs
# python trainMultWgans.py --dataset behavioral --imageSize 32 --niter 50000 --num_models 30 --experiment
# if using the tiger gpu, experiment should be in /scratch/gpfs/*username*
# if using the adroit gpu, experiment should be in /scratch/network/*username*/

import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | behavioral')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--num_models', default=30, help='Number of WGAN models to train')
parser.add_argument('--experiment', required=True, help='Where to store output data and models')
opt = parser.parse_args()


slurm_script = '''#!/bin/bash
#SBATCH --job-name=max-wgan       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)

#SBATCH --gres=gpu:2             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=mp16@princeton.edu

module purge
module load anaconda3
conda activate torch-gpu

srun python main.py --dataset {0} --dataroot ./ --imageSize {1} --nc 1 --ngf 64 --ndf 64 --niter {2} --experiment {3} --experiment_name {4} --cuda --ngpu 2 --batchSize 20 --auto_number {5}
'''

for i in range(int(opt.num_models)):

    slurm_script_name = 'train_model_{0}.slurm'.format(i)
    experiment_name = '{0}_{1}k_{2}_automated'.format(opt.dataset, opt.niter, i)

    f = open(slurm_script_name, 'w+')

    f.write(slurm_script.format(opt.dataset, opt.imageSize, opt.niter, opt.experiment, experiment_name, i))

    f.close()

    subprocess.call(['sbatch', slurm_script_name])
