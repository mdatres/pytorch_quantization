import torchvision.transforms as transforms 
import torch
import torchvision
from torchsampler import ImbalancedDatasetSampler
from PIL import Image


#Change color space
def _random_colour_space(x):
    output = x.convert("HSV")
    return output 

def check_image(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

def load_data():
    img_size = (700, 700)
    colour_transform = transforms.Lambda(lambda x: _random_colour_space(x))
    random_colour_transform = torchvision.transforms.RandomApply([colour_transform])
    normalize = torchvision.transforms.Normalize(
        mean=[1.6057, 0.0850, 0.1064], std=[1.6720, 1.1604, 0.8854])
    # train transforms
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        random_colour_transform,
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
        ])
    batch_size = 32
    test_batch_size = 4
    num_workers = 4
    balanced_dataloader = True
    # test transforms
    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        normalize,
        ])
    
    train_data_path = '/Users/massimilianodatres/si-cura_devel/si-cura_devel/cross_validation/5CV_CU_MC/CU_MC_dataset_cv/train_val'
    test_data_path = '/Users/massimilianodatres/si-cura_devel/si-cura_devel/cross_validation/5CV_CU_MC/CU_MC_dataset_cv/test'
    
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transforms, is_valid_file=check_image)
    val_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=test_transforms, is_valid_file=check_image)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transforms, is_valid_file=check_image)

    if balanced_dataloader:
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_data), num_workers=num_workers, pin_memory=True)
    else: 
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)    
    val_dataloader  = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader