
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | behavioral')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--num_models', type=int, default=30, help='Number of WGAN models to train')
parser.add_argument('--experiment', required=True, help='Where output data and models are stored')
parser.add_argument('--net_id', required=True, help='netID of the user')
opt = parser.parse_args()

for i in range(opt.num_models):

    subprocess.call(['cp', '{}/netG_{}_{}k_{}_automated.pth'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/{}/WassersteinGAN/loss_curves/netG_{}_{}k_{}_automated.pth'.format(opt.net_id, opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{}/netD_{}_{}k_{}_automated.pth'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/{}/WassersteinGAN/loss_curves/netD_{}_{}k_{}_automated.pth'.format(opt.net_id, opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{}/w_loss_{}_{}k_{}_automated.png'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/{}/WassersteinGAN/loss_curves/w_loss_{}_{}k_{}_automated.png'.format(opt.net_id, opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{}/{}_{}_training.csv'.format(opt.experiment, opt.dataset, i),
                     '/home/{}/WassersteinGAN/loss_curves/{}_{}k_{}_training.csv'.format(opt.net_id, opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{}/{}_{}_cv.csv'.format(opt.experiment, opt.dataset, i),
                     '/home/{}/WassersteinGAN/loss_curves/{}_{}k_{}_cv.csv'.format(opt.net_id, opt.dataset, opt.niter//1000, i)])
