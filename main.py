from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from matplotlib import pyplot as plt

import models.dcgan as dcgan
import models.mlp as mlp
import data.BehavioralDataset as local_dsets
from data.BehavioralHmSamples import BehavioralHmSamples

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings

if __name__=="__main__":

    warnings.filterwarnings("ignore")

    # The two calls below when run in the command line respectively run a wgan
    # with an ordinary mlp structure and a wgan with a convolutional architecture
    # on the behavioral learning dataset.
    # python main.py --dataset behavioral --dataroot ./ --imageSize 25 --nc 1 --mlp_G --mlp_D --ngf 64 --ndf 64 --niter 5
    # python main.py --dataset behavioral --dataroot ./ --imageSize 32 --nc 1 --ngf 64 --ndf 64
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nc', type=int, default=3, help='input image channels')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='number of nodes for generator hidden layers')
    parser.add_argument('--ndf', type=int, default=64, help='number of nodes for discriminator hidden layers')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
    parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01, help='lower bound for weight clipping')
    parser.add_argument('--clamp_upper', type=float, default=0.01, help='upper bound for weight clipping')
    parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
    parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
    parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
    parser.add_argument('--experiment', default=None, help='Where to store samples and models')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--experiment_name', default='', help='Tag appended to all output files')
    parser.add_argument('--auto_number', required=True, help='Tag appended to all output files when training multiple versions of one model')
    opt = parser.parse_args()
    print(opt)

    if opt.experiment is None:
        opt.experiment = os.path.join('', 'samples')
        os.system('mkdir samples')

    opt.manualSeed = random.randint(1, 10000) # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    isCnnData = not opt.mlp_G and not opt.mlp_D

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Scale(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Scale(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ])
        )
    elif opt.dataset == 'behavioral':
        tfm = transforms.Compose([transforms.ToTensor()])
        dataset = local_dsets.BehavioralDataset(isCnnData=isCnnData,
                                                auto_number=opt.auto_number,
                                                output_directory=opt.experiment)
    elif opt.dataset == 'quick':
        tfm = transforms.Compose([transforms.ToTensor()])
        dataset = BehavioralHmSamples(modelNum=3, isCnnData=isCnnData)
    assert dataset
    assert opt.batchSize <= len(dataset)
    sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=opt.batchSize)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            sampler=sampler, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = int(opt.nc)
    n_extra_layers = int(opt.n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config_{0}.json".format(opt.experiment_name)), 'w+') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt.noBN:
        netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
    elif opt.mlp_G:
        netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
    else:
        netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

    # write out generator config to generate images together wth training checkpoints (.pth)
    generator_config = {"imageSize": opt.imageSize, "nz": nz, "nc": nc, "ngf": ngf, "ngpu": ngpu, "n_extra_layers": n_extra_layers, "noBN": opt.noBN, "mlp_G": opt.mlp_G}
    with open(os.path.join(opt.experiment, "generator_config_{0}.json".format(opt.experiment_name)), 'w') as gcfg:
        gcfg.write(json.dumps(generator_config)+"\n")

    netG.apply(weights_init)
    if opt.netG != '': # load checkpoint if needed
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    if opt.mlp_D:
        netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
    else:
        netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt.adam:
        optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
        optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)

    gen_iterations = 0
    loss_d_list = []
    for epoch in range(opt.niter):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            if gen_iterations < 25 or gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt.Diters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in netD.parameters():
                    p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                batch_size = real_cpu.size(0)

                if opt.cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                real_samples = inputv

                errD_real = netD(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile = True) # totally freeze netG
                fake = Variable(netG(noisev).data)
                inputv = fake
                fake_samples = fake
                errD_fake = netD(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            # determine critic's ability to differentiate using Platt Scaling
            real_crit_scores = netD.pred(real_samples)
            fake_crit_scores = netD.pred(fake_samples)
            real_crit_scores = real_crit_scores.reshape(real_crit_scores.shape[0],1)
            fake_crit_scores = fake_crit_scores.reshape(fake_crit_scores.shape[0],1)

            x_dat = np.vstack((real_crit_scores, fake_crit_scores))
            x_dat = x_dat.astype('float64')
            y_dat = np.vstack((np.ones((real_crit_scores.shape[0],1)), -1*(np.ones((fake_crit_scores.shape[0],1)))))

            clf = LogisticRegression(solver='lbfgs')
            clf.fit(x_dat, y_dat)
            pred_labels = clf.predict(x_dat)
            batch_size = y_dat.shape[0]
            half_batch_size = batch_size // 2
            pred_labels = pred_labels.reshape(batch_size,1)
            num_correct_total = np.sum(pred_labels == y_dat)
            num_correct_real = np.sum(pred_labels[:half_batch_size, :] == y_dat[:half_batch_size, :])
            num_correct_fake = np.sum(pred_labels[half_batch_size:,:] == y_dat[half_batch_size:,:])
            print('Critic correctly identified {0} out of {1}'.format(num_correct_total, x_dat.shape[0]))
            print('Number real samples correctly identified: {0}'.format(num_correct_real))
            print('Number fake samples correctly identified: {0}'.format(num_correct_fake))
            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1

            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                % (epoch, opt.niter, i, len(dataloader), gen_iterations,
                errD.data[0], errG.data[0], errD_real.data[0], errD_fake.data[0]))

            loss_d_list.append(errD.data[0])
            # if gen_iterations % 500 == 0:
            #     real_cpu = real_cpu.mul(0.5).add(0.5)
            #     vutils.save_image(real_cpu, '{0}/real_samples.png'.format(opt.experiment))
            #     fake = netG(Variable(fixed_noise, volatile=True))
            #     fake.data = fake.data.mul(0.5).add(0.5)
            #     vutils.save_image(fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, gen_iterations))



    # save networks
    torch.save(netG.state_dict(), '{0}/netG_{1}.pth'.format(opt.experiment, opt.experiment_name))
    torch.save(netD.state_dict(), '{0}/netD_{1}.pth'.format(opt.experiment, opt.experiment_name))

    # make plot of loss_D
    plt.plot(np.arange(1, len(loss_d_list) + 1), loss_d_list)
    plt.savefig('{0}/w_loss_{1}.png'.format(opt.experiment, opt.experiment_name))
