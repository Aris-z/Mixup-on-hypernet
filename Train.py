import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import argparse
import torch.optim as optim
from utils import *
from model import model
from WeightGeneration import Weight_Generation
from CNNBlock import CNNBlock
import random
device_ids = [0,1,2,3]
# Load data
batch_size = 256

trainloader, testloader, classes = load_data(batch_size, device_ids)
###
# parser = argparse.ArgumentParser(description="Mixup on CNN demo(CIFAR10)")
# parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# args = parser.parse_args()
###

best_accuracy = 0.

weight_net1 = Weight_Generation(device_ids).cuda(device=device_ids[0])
weight_net2 = Weight_Generation(device_ids).cuda(device=device_ids[0])
# if args.resume:
#     ckpt = torch.load('./hypernetworks_cifar_paper.pth')
#     net.load_state_dict(ckpt['net'])
#     best_accuracy = ckpt['acc']

# device = torch.device(" cuda " if torch. cuda.is_available () else "cpu")
# net.to(device)

# 提前预设一部分， 后面用argparse换掉
mix_alpha = 0.1
learning_rate = 0.002
weight_decay = 0.005
milestones = [168000, 336000, 400000, 450000, 550000, 600000]
max_iter = 2000000

# multi-GPU
net = model(weight_net1, weight_net2, mix_alpha)
net = torch.nn.DataParallel(net, device_ids=device_ids)
net = net.cuda(device=device_ids[0])

optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)

criterion = nn.CrossEntropyLoss()

total_iter = 0
epochs = 0
print_freq = 50
print("Start...")
while total_iter < max_iter:
    running_loss = 0.0
    inputs_list = []
    labels_list = []
    mix_flag = True
    layer_mix = None  # 每一个epoch都随机选择一层进行mixup
    for i, data in enumerate(trainloader, 1):
        inputs, labels = data
        if inputs.shape[0] % 2 != 0:
            mix_label = None
        # inputs, labels = torch.randn(batch_size, 3, 32, 32).cuda(), torch.ones(batch_size).to(torch.int64).cuda()
        inputs, labels = Variable(inputs.cuda(device=device_ids[0])), Variable(labels.cuda(device=device_ids[0]))

        """令batch_size为偶数, 把一个batch_size对半拆"""
        if mix_flag:
            x1, x2 = torch.chunk(inputs, 2, dim=0)
            inputs_list.append(x1)
            inputs_list.append(x2)

            label1, label2 = torch.chunk(labels, 2, dim=0)
            labels_list.append(label1)
            labels_list.append(label2)
            optimizer.zero_grad()

            outputs, lam = net(inputs_list[0], inputs_list[1], batch_size, layer_mix=layer_mix)
            lam = lam[0].item()
            
            loss_func = mixup_criterion(labels_list[0], labels_list[1], lam)
            loss = loss_func(criterion, outputs)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

        # print(type(loss))
        # print(loss.data.shape)
        # print(type(loss.data[0]))
            running_loss += loss.item()
            inputs_list.clear()
            labels_list.clear()
        if not mix_flag:
            """最后一个不满足成对的batch不进行mixup"""
            inputs_list.append(inputs)
            inputs_list.append(inputs)

            labels_list.append(labels)
            labels_list.append(labels)

            optimizer.zero_grad()

            outputs, lam = net(inputs_list[0],inputs_list[1], batch_size, layer_mix=layer_mix)
            lam = lam[0].item()

            loss_func = mixup_criterion(labels_list[0], labels_list[1], lam)
            loss = loss_func(criterion, outputs)
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

        if i % print_freq == (print_freq - 1):
            print(
                "[Epoch %d, Total Iterations %6d] Loss: %.4f" % (epochs + 1, total_iter + 1, running_loss / print_freq))
            running_loss = 0.0

        total_iter += 1
    epochs += 1

    correct = 0.
    total = 0.
    for test_data in testloader:
        test_images, test_labels = test_data
        test_images = Variable(test_images.cuda(device=device_ids[0]))
        test_labels_list = [test_labels, test_labels]

        test_outputs, __ = net(test_images, test_images, batch_size)
        _, predicted = torch.max(test_outputs.cpu().data, 1)
        total += test_labels.size(0)
        correct += (predicted == test_labels).sum()

    accuracy = (100. * correct) / total
    print('After epoch %d, accuracy: %.4f %%' % (epochs, accuracy))

    if accuracy > best_accuracy:
        print('Saving model...')
        state = {
            'net': net.state_dict(),
            'acc': accuracy
        }
        torch.save(state, './hypernetworks_cifar_paper.pth')
        best_accuracy = accuracy

print('Finished Training')


