from torchvision.transforms import Normalize, Resize
from .CovidDataset import CovidDataset
import torch
import torchvision

Covid19Stats = \
    {
        'normalise':
            {
                'mean': (0.6409,),
                'std':  (0.2725,)
            },
        'quantise':
            {
                'min':   -2.351987361907959,  # computed on the normalised images of the validation partition
                'max':   1.3177745342254639,   # computed on the normalised images of the validation partition
                'scale': 0.018519585526834324
            }
    }
class Covid19Normalise(Normalize):
    def __init__(self):
        super(Covid19Normalise, self).__init__(**Covid19Stats['normalise'])



def load_data():
    transforms = []
    transforms += [Resize((96,96)), Covid19Normalise()]
    transforms = torchvision.transforms.Compose(transforms)
    train_ann = '/Users/massimilianodatres/MinimalDataset/Covid19/train/labels.csv'
    train_data = '/Users/massimilianodatres/MinimalDataset/Covid19/train/images'
    val_ann = '/Users/massimilianodatres/MinimalDataset/Covid19/val/labels.csv'
    val_data = '/Users/massimilianodatres/MinimalDataset/Covid19/val/images'
    test_data = '/Users/massimilianodatres/MinimalDataset/Covid19/test/images'
    test_ann = '/Users/massimilianodatres/MinimalDataset/Covid19/test/labels.csv'
    val_dataset = CovidDataset(annotations_file=val_ann, img_dir=val_data, transform = transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    train_dataset = CovidDataset(annotations_file=train_ann, img_dir=train_data, transform = transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle = True)
    test_dataset = CovidDataset(annotations_file=test_ann, img_dir=test_data, transform = transforms)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True)

    return train_dataloader, val_dataloader, test_dataloader