#python program to classify the handwritten digits (MNIST Dataset) using pytorch and GPU

import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision #importing the dataset
import torchvision.transforms as transforms #transform the dataset to torch.tensor
import torch.optim as optim # for use of optim algorithms in backprop
import torch.nn as nn #to define a neural network in pytorch


#loading the data
batch_size = 128 
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())#transforms convert raw image downloaded into tensor for pytorch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

#defining the class for the CNN
class LeNet(nn.Module):
    def __init__(self): 
        super(LeNet, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(1, 6, 5),         # (N, 3, 32, 32) -> (N,  6, 28, 28)
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),  # (N, 6, 28, 28) -> (N,  6, 14, 14)
            nn.Conv2d(6, 16, 5),        # (N, 6, 14, 14) -> (N, 16, 10, 10)  
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)   # (N,16, 10, 10) -> (N, 16, 5, 5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(256,120),         # (N, 400) -> (N, 120)
            nn.ReLU(),
            nn.Linear(120,84),          # (N, 120) -> (N, 84)
            nn.ReLU(),
            nn.Linear(84,20),           # (N, 84)  -> (N, 10)
            nn.ReLU(),
            nn.Linear(20,10)
        )
        
    def forward(self, x): # define forward propogation for CNN
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1) # After CNN operation convert images into 1D array to fed to Neural network
        x = self.fc_model(x)
        return x #return output
      
    def evaluation(self,dataloader): # to calculate accuracy
        total, correct = 0, 0
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) #move inputs and labels to gpu
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item() #check correct predictions 
        return 100 * correct / total  #return accuracy

    def fit(self,trainloader, opt, max_epoch=10):
        
        for epoch in range(max_epoch):
          for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad() # make all the gradients of optim to zero

            outputs = self.forward(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward() #calculate the derivatives in pytorch
            opt.step() #take a set of GD according to the optim
        
          print('Epoch: %d/%d' % (epoch, max_epoch))
    

#check if gpu is availabe and use it    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = LeNet().to(device) #class instantiation
loss_fn = nn.CrossEntropyLoss() #define loss function
opt = optim.Adam(net.parameters(),lr=1e-3, weight_decay=1e-4) #set optim agorithm and its parameters
net.fit(trainloader,opt,30) #fit the data

print('Test acc: %0.2f, Train acc: %0.2f' % (net.evaluation(testloader), net.evaluation(trainloader))) #check accuracy
