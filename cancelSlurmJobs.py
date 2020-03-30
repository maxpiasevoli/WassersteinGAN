
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--first_id', type=int, required=True)
parser.add_argument('--num_jobs', type=int, default=30, help='the height / width of the input image to network')
opt = parser.parse_args()

for i in range(opt.num_jobs):

    subprocess.call(['scancel', str(opt.first_id + i)])
