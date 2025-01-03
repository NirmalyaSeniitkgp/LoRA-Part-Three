# Implementation of LoRA (Low-Rank Adaptation)
# This is Part Three of My Explanation
# This Program gives basic understanding of how
# LoRA is implemented in a Model.
# Here, Model is a Sequential Connection of Two Layers.
# Layers are connected by using nn.ModuleList()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
from torch.utils.data import DataLoader
import torch.nn.utils.parametrize as parametrize

# Hyper Parameters
num_epochs = 1
batch_size = 50
learning_rate = 0.001
number_of_layers = 2
num_classes = 10

# Make PyTorch Deterministic
_=torch.manual_seed(0)

# MNIST Data Set
train_dataset = torchvision.datasets.MNIST(root = '/home/idrbt-06/Desktop/PY_TORCH/Feed_Forward_Neural_Net/Data', train = True,
        transform = transforms.ToTensor(), download = True)

test_dataset = torchvision.datasets.MNIST(root = '/home/idrbt-06/Desktop/PY_TORCH/Feed_Forward_Neural_Net/Data', train = False,
        transform = transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = False)

test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

# Implementation of NeuralNet
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SequentialLayersGeneration(nn.Module):
    def __init__(self,input_size, hidden_size_1, hidden_size_2, number_of_layers):
        super().__init__()
        self.layers = nn.ModuleList([NeuralNet(input_size, hidden_size_1, hidden_size_2)
                                                    for i in range(number_of_layers)])

    def forward(self, x):
        for i in self.layers:
            x = i(x)
        return x


class SequentialNeuralNet(nn.Module):
    def __init__(self,input_size, hidden_size_1, hidden_size_2,number_of_layers,num_classes):
        super().__init__()
        self.sequential_layers = SequentialLayersGeneration(input_size,hidden_size_1,hidden_size_2,number_of_layers)
        self.mlp_head = nn.Linear(input_size,num_classes)
    
    def forward(self, x):
        x = self.sequential_layers(x)
        x = self.mlp_head(x)
        return x


model = SequentialNeuralNet(input_size=28*28,
                            hidden_size_1=1000,
                            hidden_size_2=2000,
                            number_of_layers=2,
                            num_classes=10)

# Definition of Training Function
def Training(model,train_loader,learning_rate,num_epochs,Total_num_steps=None):
    model.train()
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    # Training Loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images
            labels = labels
            # Forward Pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss = total_loss + loss.item()
            # Backward and Update
            loss.backward()
            optimizer.step()
            # Total Number of Steps Calculation
            if Total_num_steps is not None:
                Total_steps = (epoch+1)*(i+1)
                if Total_steps>=Total_num_steps:
                    return
            # print information
        average_loss = total_loss/(i+1)
    print('Average Loss =',average_loss)

# Calling the Training Function
Training(model,train_loader,learning_rate,num_epochs)

# Display the original Weights
print('*'*30)
print('BEFORE LoRA APPLICATION')
for name, param in model.named_parameters():
    print('Layer Name =',name)
    print('Parameter Size =',param.shape)


# Total Number of Parameters
print('*'*30)
total_number_of_parameters = 0
for param in model.parameters():
    total_number_of_parameters += param.numel()
print('Total Number of Parameters =',total_number_of_parameters)

# Definition of Test Function
def Testing(model,test_loader):
    model.eval()
    # Test Loop
    with torch.no_grad():
        number_correct = 0
        number_samples = 0
        wrong_counts = [0 for i in range(10)]
        for images, labels in test_loader:
            images =images
            labels = labels
            outputs = model(images)
            # Calculation of Prediction
            predictions = torch.argmax(outputs, 1)
            # Calculation of Total Accuracy
            number_samples +=labels.shape[0]
            number_correct +=(predictions==labels).sum().item()
            # Calculation of Digit Wise Error
            for i in range(len(predictions)):
                if predictions[i]!= labels[i]:
                    wrong_counts[labels[i]]=wrong_counts[labels[i]]+1

    print('*'*30)
    acc = (number_correct/number_samples)*100.0
    print(f'accuracy={acc}')
    print('*'*30)

    for i in range(num_classes):
        print(f'Wrong Count for Digit {i} = {wrong_counts[i]}')

# Calling the Test Function
# Accuracy Before Application of LoRA
print('Accuracy Before Application of LoRA')
Testing(model,test_loader)

# Implementing LoRA Logic
class LoRAParameterization(nn.Module):
    def __init__(self,in_features,out_features,rank,alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank,out_features))
        self.lora_B = nn.Parameter(torch.zeros(in_features,rank))
        nn.init.normal_(self.lora_A, mean=0, std=1.0)
        self.scale = alpha/rank
    
    def forward(self,original_weights):
        return original_weights + (torch.matmul(self.lora_B,self.lora_A))*self.scale


# Implementing the Rapper Function of LoRA Logic
# This function returns the object of class LoRAParameterization
# Here, the Returned Object p is Callable
def linear_layer_parameterization(layer,rank=1,alpha=1):
    (in_features,out_features) = layer.weight.shape
    p = LoRAParameterization(in_features,out_features,rank,alpha)
    return p


# Implementing the LoRA Parameterization in our Model
parametrize.register_parametrization(model.sequential_layers.layers._modules['0'].linear1,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['0'].linear1))
parametrize.register_parametrization(model.sequential_layers.layers._modules['0'].linear2,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['0'].linear2))
parametrize.register_parametrization(model.sequential_layers.layers._modules['0'].linear3,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['0'].linear3))
parametrize.register_parametrization(model.sequential_layers.layers._modules['1'].linear1,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['1'].linear1))
parametrize.register_parametrization(model.sequential_layers.layers._modules['1'].linear2,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['1'].linear2))
parametrize.register_parametrization(model.sequential_layers.layers._modules['1'].linear3,'weight',linear_layer_parameterization(model.sequential_layers.layers._modules['1'].linear3))

print('*'*30)
print('AFTER LoRA APPLICATION')
for name, param in model.named_parameters():
    print('Layer Name',name)
    print('Parameter Size',param.shape)

print('*'*30)
print('PARAMETERS WHERE LoRA NOT IN NAME')
for name, param in model.named_parameters():
    if 'lora' not in name:
        print(f'Layer Name = {name}')
        print(f'Layer Shape = {param.shape}')

print('*'*30)
print('PARAMETERS WHERE LoRA IN NAME')
for name, param in model.named_parameters():
    if 'lora' in name:
        print(f'Layer Name = {name}')
        print(f'Layer Shape = {param.shape}')

print('*'*30)
print('INCREMENT OF PARAMETERS AFTER APPLICATION OF LoRA')
total_non_lora_parameters = 0
for name, param in model.named_parameters():
    if 'lora' not in name:
        total_non_lora_parameters += param.numel()
print('Non LoRA Parameters =',total_non_lora_parameters)

total_lora_parameters = 0
for name, param in model.named_parameters():
    if 'lora' in name:
        total_lora_parameters += param.numel()
print('LoRA Parameters =',total_lora_parameters)

parameters_increment_due_to_LoRA = (total_lora_parameters/total_non_lora_parameters)*100
print('% Parameters Increment Due to LoRA',parameters_increment_due_to_LoRA)

# Seperating the Digit 3 from Original MNIST Training Data Set
required_index = train_dataset.targets == 3

train_dataset.data = train_dataset.data[required_index]

train_dataset.targets = train_dataset.targets[required_index]

train_loader_3 = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

# Freeze the Gradients of NON LoRA Parameters
for name, param in model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# Calling the Training Function
# Here, We are Fine Tuning the Digit 3
Training(model,train_loader_3,learning_rate,num_epochs,Total_num_steps=30)
# Calling the Test Function after Fine Tuning the Digit 3
Testing(model,test_loader)



