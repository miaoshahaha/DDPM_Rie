import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, Subset

def load_dataset(curr_dataset='CIFAR10',):

    DatasetTransform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.RandomCrop((26,26)),
        transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2),
        #transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        

    if curr_dataset == 'CIFAR10':
        train_dataset = CIFAR10(root='./CIFAR10_ds', train=True, download=True, transform=DatasetTransform)
        test_dataset = CIFAR10(root='./CIFAR10_ds', train=False, download=True, transform=DatasetTransform)
        #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    elif curr_dataset == 'ImageNet':
        train_dataset = torchvision.datasets.ImageFolder(root='imagenet-mini/train', transform=DatasetTransform)
        test_dataset = torchvision.datasets.ImageFolder(root='imagenet-mini/val', transform=DatasetTransform)
        #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        #test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    elif curr_dataset == 'CelebA':
        train_dataset = torchvision.datasets.ImageFolder(root='celeba/train', transform=DatasetTransform)
        test_dataset = torchvision.datasets.ImageFolder(root='celeba/val', transform=DatasetTransform)
    
    else:
        SyntaxError('Unrecongnize dataaset.')


    return train_dataset, test_dataset
