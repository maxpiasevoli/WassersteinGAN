
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | behavioral')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--num_models', default=30, help='Number of WGAN models to train')
parser.add_argument('--experiment', required=True, help='Where output data and models are stored')
opt = parser.parse_args()

for i in range(opt.num_models):

    subprocess.call(['cp', '{0}/netG_{1}_{2}k_{3}_automated.pth'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/netG_{0}_{1}k_{2}_automated.pth'.format(opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{0}/netD_{1}_{2}k_{3}_automated.pth'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/netD_{0}_{1}k_{2}_automated.pth'.format(opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{0}/w_loss_{1}_{2}k_{3}_automated.png'.format(opt.experiment, opt.dataset, opt.niter, i),
                     '/home/mp16/WassersteinGAN/loss_curves/w_loss_{0}_{1}k_{2}_automated.png'.format(opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{0}/{1}_{2}_training.csv'.format(opt.experiment, opt.dataset, i),
                     '/home/mp16/WassersteinGAN/loss_curves/{0}_{1}k_{2}_training.csv'.format(opt.dataset, opt.niter//1000, i)])

    subprocess.call(['cp', '{0}/{1}_{2}_cv.csv'.format(opt.experiment, opt.dataset, i),
                     '/home/mp16/WassersteinGAN/loss_curves/{0}_{1}k_{2}_cv.csv'.format(opt.dataset, opt.niter//1000, i)])
