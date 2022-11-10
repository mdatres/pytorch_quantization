import torch
from pytorch_quantization.observers import PACTFakeQuantize

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