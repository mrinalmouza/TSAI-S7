import torch
from torchvision import datasets, transforms


# training_transform= transforms.Compose([transforms.ToTensor(),
#                                         transforms.Normalize((0.1307,), (0.3081,))])

# testing_transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))])


# training_data = datasets.MNIST('../data',train = True, transform = training_transform, download =True)

# testing_data = datasets.MNIST('../data', train = False, transform = testing_transform, download =True)



# batch_size = 128

# kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}
# # kwargs = dict(shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True)

# train_loader = torch.utils.data.DataLoader(training_data, **kwargs)

# test_loader = torch.utils.data.DataLoader(testing_data, **kwargs)

# Train Phase transformations
train_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.RandomRotation((-15., 15.), fill=0),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,)) # The mean and std have to be sequences (e.g., tuples), therefore you should add a comma after the values. 
                                       # Note the difference between (0.1307) and (0.1307,)
                                       ])

# Test Phase transformations
test_transforms = transforms.Compose([
                                      #  transforms.Resize((28, 28)),
                                      #  transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

train = datasets.MNIST('./data', train=True, download=True, transform=train_transforms)
test = datasets.MNIST('./data', train=False, download=True, transform=test_transforms)

# dataloader arguments - something you'll fetch these from cmdprmt
dataloader_args = dict(shuffle=True, batch_size=128, num_workers=4, pin_memory=True)

# train dataloader
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)

# test dataloader
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)



