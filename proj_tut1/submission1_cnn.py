import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            #TODO: add 4 sub-layers, 
            # Conv2d, parameters: in_channels=1, out_channels=16. kernel_size=5, stride=1, padding=2
            # BatchNorm2d, parameters: num_features=16
            # ReLU(), 
            # MaxPool2d, parameters: kernel_size=2, stride=2
            # =================================================
            # coding here:
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            #TODO: add 4 sub-layers, 
            # Conv2d, parameters: in_channels=16, out_channels=32. kernel_size=5, stride=1, padding=2
            # BatchNorm2d, parameters: num_features=32
            # ReLU(), 
            # MaxPool2d, parameters: kernel_size=2, stride=2
            #=======================================
            # coding here:
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            #TODO: add 3 sub-layers, 
            # Conv2d, parameters: in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2
            # BatchNorm2d, parameters: num_features=64
            # ReLU()
            #=======================================
            # coding here:
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.fc = nn.Linear(7*7*64, num_classes)
        
    def forward(self, input):
        # TODO
        # layers are stacked in following sequence
        # x-> self.layer1 -> self.layer2 -> self.layer3 -> squeeze -> self.fc -> out
        #================================================================
        # coding here:
        # layer 1
        output = self.layer1(input)
        # layer 2
        output = self.layer2(output)
        output = self.layer3(output)
        output = output.reshape(output.size(0), -1) # squeeze from 4D array to 2D array
        # layer fc
        output = self.fc(output)
        return output

model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # TODO: 
        # use model for prediction and calculate loss
        # the signals are flowed in the following sequence
        # images -> class ConvNet() -> outputs
        # (outputs, labels) -> class CrossEntropyLoss() -> loss
        #=======================================================
        # coding here
        outputs = model(images) # modify None to the correct answer
        loss = criterion(outputs, labels) # modify None to the correct answer

        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')




#=====================================================
# the console will print out accuracy result:
#   Test Accuracy of the model on the 10000 test images: 99.2 %
# 
# Number of accuracy should be similar. If not, you might need to check your code.