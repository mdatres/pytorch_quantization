from multiprocessing.spawn import import_main_path
import torch 
import math
import os
from torch.utils.tensorboard import SummaryWriter
from __init__ import PACTFakeQuantize
from meters import AverageMeter, accuracy
def get_children(model: torch.nn.Module):
   
        children = list(model.children())
        flatt_children = []
        if children == []:
            # if model has no children; model is last child! :O
            return model
        else:
        # look for children from children... to the last child!
            for child in children:
                    if not isinstance(child, PACTFakeQuantize):
                        try:
                            flatt_children.extend(get_children(child))
                        except TypeError:
                            flatt_children.append(get_children(child))

                    else: 
                        flatt_children.append(child)

        return flatt_children

def test(model, criterion, dataloader, device) -> float:
    correct = 0
    total = 0
    loss = 0
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            
            inputs, labels = data

            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total, loss / len(dataloader)
    

