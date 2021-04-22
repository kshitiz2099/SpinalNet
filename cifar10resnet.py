import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

"""**Loading Data and Preprocessing**"""

transforms=transforms.Compose([
                               transforms.RandomCrop(32),
                               transforms.Pad(4),
                               transforms.RandomHorizontalFlip(),
                               transforms.RandomAffine(0,translate=(0.2,0.2), shear=(0.2,0.2,0.2,0.2)),
                               transforms.ToTensor()          
])

trainset = CIFAR10(root='data/', download=True, transform=transforms)
testset = CIFAR10(root='data/', train=False, transform=ToTensor())

layer_width=20
half_width=256
batch_size=100

trainloader=DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
testloader=DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""# **Models and Utils**

**ResNet18**
"""

resnet18=torchvision.models.resnet18(pretrained=True)

resnet18.fc=nn.Sequential(
      nn.Dropout(0.1),
      nn.Linear(512, 10)
)

resnet18.to(device)


"""**SpinalNet**"""
class SpinalNet(nn.Module):
  def __init__(self,resnet):
    super().__init__()
    self.resnet=resnet
    self.fc1=nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(half_width,layer_width),
        nn.ReLU(inplace=True)
    )

    self.fc2=nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(half_width+layer_width,layer_width),
        nn.ReLU(inplace=True)
    )

    self.fc3=nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(half_width+layer_width,layer_width),
        nn.ReLU(inplace=True)
    )

    self.fc4=nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(half_width+layer_width,layer_width),
        nn.ReLU(inplace=True)
    )

    self.out=nn.Sequential(
        nn.Dropout(0.25),
        nn.Linear(layer_width*4,10)
    )

  def forward(self, x):
    x=self.resnet(x)
    x=x.view(x.size()[0], -1)
    x1=self.fc1(x[:,0:half_width])
    x2=self.fc2(torch.cat([x[:,half_width:half_width*2], x1], dim=1))
    x3=self.fc3(torch.cat([x[:,0:half_width], x2], dim=1))
    x4=self.fc4(torch.cat([x[:,half_width:half_width*2], x3], dim=1))

    x5=torch.cat([x1, x2], dim=1)
    x5=torch.cat([x5, x3], dim=1)
    x5=torch.cat([x5, x4], dim=1)

    output=self.out(x5)

    return x5

#Remove the top layer of resnet to use the rest with SpinalNet layers
def removeTop(resnet):
  layers=[]
  i=0
  for layer in resnet.children():
    layers+=[layer]
    i=i+1
    if(i==9):
      break;
  return nn.Sequential(*layers)

resnet=removeTop(resnet18)

snet=SpinalNet(resnet).to(device)

loss_function=nn.CrossEntropyLoss()
optimizer1=optim.Adam(snet.parameters(), lr=0.0001)
optimizer2=optim.Adam(resnet18.parameters(), lr=0.0001)
snet_scheduler=StepLR(optimizer1, step_size=30, gamma=0.5)
resnet_scheduler=StepLR(optimizer2, step_size=30, gamma=0.5)
snet_history={'loss':[], 'accuracy':[]}
resnet18_history={'loss':[], 'accuracy':[]}

epochs=150

"""# **Training**"""

for epoch in range(epochs):
  snet_running_loss=0
  resnet18_running_loss=0

  print("epoch ", epoch+1, "/", epochs)
  for images, labels in tqdm_notebook(trainloader):
    images=images.to(device)
    labels=labels.to(device)

    snet_out=snet(images)
    snet_loss=loss_function(snet_out, labels)
    snet_running_loss+=snet_loss
    optimizer1.zero_grad()
    snet_loss.backward()
    optimizer1.step()

    resnet18_out=resnet18(images)
    resnet18_loss=loss_function(resnet18_out, labels)
    resnet18_running_loss+=resnet18_loss
    optimizer2.zero_grad()
    resnet18_loss.backward()
    optimizer2.step()

  snet_scheduler.step()
  resnet_scheduler.step()

  snet.eval()
  resnet18.eval()

  snet_history['loss']+=[snet_running_loss/len(trainloader)]
  resnet18_history['loss']+=[resnet18_running_loss/len(trainloader)]

  with torch.no_grad():
    correct_snet=0
    correct_resnet18=0

    total=0

    for images, labels in testloader:
      images=images.to(device)
      labels=labels.to(device)

      snet_output=snet(images)
      resnet18_output=resnet18(images)

      total+=labels.size(0)

      _,predictions_snet=torch.max(snet_output.data, 1)
      _,predictions_resnet18=torch.max(resnet18_output.data, 1)
      correct_snet+=(predictions_snet==labels).sum().item()
      correct_resnet18+=(predictions_resnet18==labels).sum().item()

    snet.train()
    resnet18.train()

  print("SpinalNet Accuracy:", correct_snet/total)
  snet_history['accuracy']+=[correct_snet/total]

  print("ResNet18 Accuracy:", correct_resnet18/total)
  resnet18_history['accuracy']+=[correct_resnet18/total]

"""### **Save Model Weights and History**"""

snet_hist=open('/SpinalNetResnet18.txt', 'w')
snet_hist.write(str(snet_history))
snet_hist.close()

resnet18_hist=open('/Resnet18Normal.txt', 'w')
resnet18_hist.write(str(resnet18_history))
resnet_hist.close()

torch.save(snet.state_dict(), '/SpinalResnetc10.pth')
torch.save(snet.state_dict(), '/Resnet18c10.pth')