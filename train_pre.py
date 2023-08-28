import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from dcgan import Discriminator, Generator, weights_init
from dataset import Dataset
from dataset2 import Dataset2
import os


lrD = 1e-5
lrG = 1e-4
beta1 = 0.5
start_epoch =0
epoch_num = 200
batch_size = 32
nz = 100  # length of noise
ngpu = 2
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    noise_down = -1
    noise_up = 1  
    trainset = Dataset2('./sdss_data/datasetall')

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )

    # init netD and netG
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    netG = Generator(nz).to(device)
    netG.apply(weights_init)

    netD.to(device)
    netG.to(device)
    criterion = nn.BCELoss()

    fixed_noise = torch.rand(16, nz, 1, device=device) * (noise_up - noise_down) + noise_down

    real_label = 1.
    fake_label = 0.

    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    errD_list=[]
    errG_list=[]
    errTotal_list=[]
    errDM_list=[]
    errGM_list=[]
    errTotalM_list=[]

    for epoch in range(start_epoch,  epoch_num):
        count = 0
        for step, (data, target) in enumerate(trainloader):

            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label,
                               dtype=torch.float, device=device)
            optimizerD.zero_grad()
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.rand(b_size, nz, 1, device=device) * (noise_up - noise_down) + noise_down
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            errD_list.append(errD)
            optimizerD.step()
            # netG.zero_grad()
            optimizerG.zero_grad()

            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG_list.append(errG)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            errTotal = errD+errG
            errTotal_list.append(errTotal)

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_T:%.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epoch_num, step, len(trainloader),
                     errD.item(), errG.item(),errTotal.item(), D_x, D_G_z1, D_G_z2))
            torch.cuda.empty_cache()
        errD_array = np.array([x.detach().cpu().numpy() for x in errD_list])
        errG_array = np.array([x.detach().cpu().numpy() for x in errG_list])
        errDM_list.append(torch.stack(errD_list).mean().item())
        errGM_list.append(torch.stack(errG_list).mean().item())
        errTotalM_list.append(torch.stack(errTotal_list).mean().item())

        errD = errDM_list[-1]
        errG = errGM_list[-1]

        print('[%d]epoch\tLoss_DM:%.4f\tLoss_GM:%.4f' % (epoch, errD, errG))
        with open('./GAN_new_log/saveGAN2/train_lognew1.txt', 'w') as f:
          for epoch, errD, errG ,errT in zip(range(start_epoch, epoch + 1), errDM_list, errGM_list,errTotalM_list):
            f.write(f'Epoch {epoch}: Loss_DM: {errD:.4f}, Loss_GM: {errG:.4f}, Loss_TM:{errT:.4f}\n')

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
            f, a = plt.subplots(4, 4, figsize=(8, 8))
            for i in range(4):
                for j in range(4):
                    a[i][j].plot(fake[i * 4 + j].view(-1))
                    a[i][j].set_xticks(())
                    a[i][j].set_yticks(())
            np.save('./GAN_data/Gdata_{}'.format(epoch),fake)
            plt.savefig('./GAN_new_log/stellargan_epoch_%d.png' % epoch)
            plt.close()
        # save models
        torch.save(netG.state_dict(), './GAN_new_log/stellargan/stellarganG_%d.pkl' % epoch)
        torch.save(netD.state_dict(), './GAN_new_log/stellargan/stellarganD__%d.pkl' % epoch) 




    

if __name__ == '__main__':
    main()