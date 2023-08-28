import torch
import torch.nn as nn
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from cnn import Cnn, weights_init
from dataset import Dataset
from dataset1 import Dataset1
from dataset2 import Dataset2
from sklearn.metrics import f1_score

lr = 2e-5
beta1 = 0.5
epoch_num = 200
batch_size = 16
nz = 100 
ngpu = 2
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")





def main():
    trainset = Dataset2('./sdss_data/dataset/train')


    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=16, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=True
    )

    netC = Cnn().to(device)

    print(netC)
    torch.cuda.empty_cache() 

    state_dict = torch.load('./nets/dcgan_netD.pkl', map_location={'cuda:2':'cuda:3'})
    netC.load_state_dict(state_dict,strict=False)

    criteon = nn.CrossEntropyLoss()


    optimizerC = optim.Adam(netC.parameters(), lr=lr, betas=(beta1, 0.999))

    loss_list=[]
    loss_all_list = []
    f1_list = []
    f1_all_list = []
    f1_vlist = []
    f1_all_vlist = []
    f1_tlist = []
    f1_all_tlist = []
    f1_scores = {0: 0.19999999999999998, 1: 0.0, 2: 0.22222222222222224, 3: 0.0, 4: 0.0, 5: 0.4117647058823529, 6: 0.0}

    total_f1_scores = {class_label: 0.0 for class_label in f1_scores.keys()}

    total_f1_scoresT = {class_label: 0.0 for class_label in f1_scores.keys()}

    all_average_f1_scores = []



    for epoch in range(epoch_num):
        count = 0
        for step, (data, target) in enumerate(trainloader):
            real_cpu = data.to(device)
            netC.to(device)

            real_cpu = real_cpu.to(device)
            
            optimizerC.zero_grad()
            output = netC(real_cpu)
            pred_train = output.argmax(dim=1)
            output.to(device)
            target = target.long()  
            target=target.to(device)
            criteon.to(device)


            loss_train = criteon(output, target)


            f1_s=f1_score(target.cpu(), pred_train.cpu(), average='macro')
            loss_train.backward()
            optimizerC.step()
            print('[%d/%d][%d/%d]\tLoss_T: %.4f\tLoss_G: %.4f\tF1_score: %.4f'
                  % (epoch, epoch_num, step, len(trainloader),
                     loss_train.item(),loss_train.item(),f1_s))
            loss_list.append(loss_train)
            f1_list.append(f1_s)
            torch.cuda.empty_cache()

        if(epoch==198):
                with torch.no_grad():
                    testloader_length = len(testloader)


                    for step, (data, target) in enumerate(testloader):
                        data = data.to(device)
                        outputs = netC(data)
                        pred_train = outputs.argmax(dim=1)

                        unique_classes = torch.unique(target)


                        f1_scores = {}

                        for class_label in range(7):

                            true_binary = (target == class_label).float()
                            pred_binary = (pred_train == class_label).float()
    

                            f1 = f1_score(true_binary.cpu(), pred_binary.cpu(),zero_division=1)
    

                            f1_scores[class_label] = f1
                        for class_label in total_f1_scores:
                            total_f1_scoresT[class_label] += f1_scores[class_label]
                        f1_ts=f1_score(target.cpu(), pred_train.cpu(), average='macro')
                        f1_tlist.append(f1_ts)

                        current_progress = (step + 1) / testloader_length * 100
                        print(f'Testing - Step: {step+1}/{testloader_length} - Progress: {current_progress:.2f}%')
                    f1_tlist_tensors = [torch.tensor(arr) for arr in f1_tlist]
                    f1_all_tlist.append(torch.stack(f1_tlist_tensors).mean().item())
                    average_f1_scores = {class_label: total_f1_scoresT[class_label] / (step+1) for class_label in total_f1_scores}
                with open('./img/cnn/test_new_GCNN0.1.txt', 'w') as f:
                    for f1_tas in zip(f1_all_tlist):
                        f.write(f'F1-scoreT:{f1_tas[0]:.4f}\n')
                        for label, f1 in average_f1_scores.items():
                            f.write(f' F1-score_{label}: {f1:.4f}')
            


        f1_list_tensors = [torch.tensor(arr) for arr in f1_list]

        loss_all_list.append(torch.stack(loss_list).mean().item())
        f1_all_list.append(torch.stack(f1_list_tensors).mean().item())

        loss = loss_all_list[-1]

        f1_as = f1_all_list[-1]

        print('[%d]epoch\tLoss_ALL:%.4f' % (epoch, loss))

        with torch.no_grad():
            valloader_length = len(valloader)
            total_f1_scores = {class_label: 0.0 for class_label in f1_scores.keys()}
            for step, (data, target) in enumerate(valloader):
                data = data.to(device)
                outputs = netC(data)
                pred_train = outputs.argmax(dim=1)



                f1_scores = {}

                for class_label in range(7):

                    true_binary = (target == class_label).float()
                    pred_binary = (pred_train == class_label).float()
    

                    f1 = f1_score(true_binary.cpu(), pred_binary.cpu(),zero_division=1)
    

                    f1_scores[class_label] = f1
                for class_label in total_f1_scores:
                    total_f1_scores[class_label] += f1_scores[class_label]
                f1_vs=f1_score(target.cpu(), pred_train.cpu(), average='macro')
                f1_vlist.append(f1_vs)
                # Calculate current progress
                current_progress = (step + 1) / valloader_length * 100
                print(f'Valing - Step: {step+1}/{valloader_length} - Progress: {current_progress:.2f}%')
                # if step==5:
                #     break
            f1_vlist_tensors = [torch.tensor(arr) for arr in f1_vlist]
            f1_all_vlist.append(torch.stack(f1_vlist_tensors).mean().item())
            average_f1_scores = {class_label: total_f1_scores[class_label] / (step+1) for class_label in total_f1_scores}
            all_average_f1_scores.append(average_f1_scores)


        with open('./img/train.txt', 'w') as f:
            for epoch, loss, f1_as, f1_vas in zip(range(0, epoch + 1), loss_all_list, f1_all_list, f1_all_vlist):
                f.write(f'Epoch {epoch}: Loss_DM: {loss:.4f}')
                for label, f1 in all_average_f1_scores[epoch].items():
                    f.write(f' F1-score_{label}: {f1:.4f}')
                f.write(f' F1-scoreV: {f1_vas:.4f}\n')
                f.write(f' F1_score: {f1_as:.4f}\n')

if __name__ == '__main__':
    main()