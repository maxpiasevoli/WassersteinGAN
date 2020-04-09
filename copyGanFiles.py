
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--num_models', default=30, help='Number of WGAN models to train')
opt = parser.parse_args()

for i in range(opt.num_models):

    subprocess.call(['cp', '/scratch/gpfs/mp16/netG_{0}k_{1}_automated.pth'.format(opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/netG_{0}k_{1}_automated.pth'.format(opt.niter//1000, i)])

    subprocess.call(['cp', '/scratch/gpfs/mp16/netD_{0}k_{1}_automated.pth'.format(opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/netD_{0}k_{1}_automated.pth'.format(opt.niter//1000, i)])

    subprocess.call(['cp', '/scratch/gpfs/mp16/w_loss_{0}k_{1}_automated.png'.format(opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/w_loss_{0}k_{1}_automated.png'.format(opt.niter//1000, i)])
