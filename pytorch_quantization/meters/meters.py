import torch
import os



def accuracy(output, target):
    """ Computes the top 1 accuracy """
    with torch.no_grad():
        batch_size = target.size(0)

        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        correct_one = correct[:1].view(-1).float().sum(0, keepdim=True)
        return correct_one.mul_(100.0 / batch_size).item()

# def print_size_of_model(model):
#     """ Prints the real size of the model """
#     torch.save(model.state_dict(), "temp.p")
#     print('Size (MB):', os.path.getsize("temp.p")/1e6)
#     os.remove('temp.p')

# def load_model(quantized_model, model):
#     """ Loads in the weights into an object meant for quantization """
#     state_dict = model.state_dict()
#     model = model.to('cpu')
#     quantized_model.load_state_dict(state_dict)

