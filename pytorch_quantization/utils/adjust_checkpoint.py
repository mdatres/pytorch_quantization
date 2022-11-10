import torch 

def adjust_checkpoint(net, path):

    ckpt_f = torch.load(path, map_location=torch.device('cpu'))
    new_f = list(ckpt_f.items())

    my_model_kvpair = net.state_dict()
    count=0
    for key,value in my_model_kvpair.items():
        
        layer_name, weights=new_f[count] 
        
        my_model_kvpair[key]= weights
        count+=1

    return my_model_kvpair

