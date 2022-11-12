import torch 

import os
from torch.quantization.qconfig import QConfig
from pytorch_quantization.utils import get_children
import json
import importlib
from torch.utils.tensorboard import SummaryWriter
from experiments.Covid19.data import load_data
from pytorch_quantization.utils import adjust_checkpoint

def train(path_to_experiment):

    config = os.path.join(path_to_experiment, 'config.json')
    try: 
        with open(config) as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        print("You need to write config.json")

    path_save = os.path.join(path_to_experiment, 'logs/saves')
    path_to_tens = os.path.join(path_to_experiment, 'logs/tensorboard')
    
    net_class = config['model']['model']
    net_kwargs = config['model']['kwargs']
    
    pyquant_models      = importlib.import_module("pytorch_quantization.models")
    try:
        net_class = getattr(pyquant_models, net_class)
        net = net_class(**net_kwargs)
    except ModuleNotFoundError:
        print("Implement your model in pytorch_quantization/models")

    train_dataloader, val_dataloader, test_dataloader = load_data()
    device = config["device"]
    
    quant = config["quant"]["quant"]
    quant_type = config["quant"]["type"]
    if quant:
        try:
            quant_observer_activations = config["quant"]["observer"]["activations"]["name"]
            quant_observer_weights = config["quant"]["observer"]["weights"]["name"]
            quant_observer_args_activations = config["quant"]["observer"]["activations"]["kwargs"]
            quant_observer_args_weights = config["quant"]["observer"]["weights"]["kwargs"]
        except:
            try:
                quant_observer_default = config["quant"]["observer"]["observer"]
                quant_observer_default_args = config["quant"]["observer"]["kwargs"]
            except:
                if quant_type== 'qat':
                    quant_observer_default = "default_qat_qconfig"
                elif quant_type== 'static':
                    quant_observer_default = "default_qconfig"
    
    
        pyquant_observers      = importlib.import_module("pytorch_quantization.observers")
        try: 
            observer_act_class = getattr(pyquant_observers, quant_observer_activations)
            quant_observer_args_activations["dtype"] = getattr(torch, quant_observer_args_activations["dtype"])
            quant_observer_args_activations["qscheme"] = getattr(torch, quant_observer_args_activations["qscheme"])
            observer_weights_class = getattr(pyquant_observers, quant_observer_weights)
            quant_observer_args_weights["dtype"] = getattr(torch, quant_observer_args_weights["dtype"])
            quant_observer_args_weights["qscheme"] = getattr(torch, quant_observer_args_weights["qscheme"])
            qconfig = QConfig(activation=observer_act_class.with_args(**quant_observer_args_activations), weight =observer_weights_class.with_args(**quant_observer_args_weights))  
        except: 
            try:
                py_observers = importlib.import_module("torch.quantization.observers")
                observer_act_class = getattr(py_observers, quant_observer_activations)
                observer_weights_class = getattr(py_observers, quant_observer_weights)
                quant_observer_args_activations["dtype"] = getattr(torch, quant_observer_args_activations["dtype"])
                quant_observer_args_activations["qscheme"] = getattr(torch, quant_observer_args_activations["qscheme"])
                observer_weights_class = getattr(pyquant_observers, quant_observer_activations)
                quant_observer_args_weights["dtype"] = getattr(torch, quant_observer_args_weights["dtype"])
                quant_observer_args_weights["qscheme"] = getattr(torch, quant_observer_args_weights["qscheme"])
                qconfig = QConfig(activation=observer_act_class.with_args(**quant_observer_args_activations),
                            weight =observer_weights_class.with_args(**quant_observer_args_weights))
            except:
                try:
                    py_observers = importlib.import_module("torch.ao.quantization")
                    observer_weights_class = getattr(py_observers, quant_observer_default)
                    qconfig = observer_weights_class(quant_observer_default_args)
                except:
                    try:
                        py_observers = importlib.import_module("torch.ao.quantization")
                        observer_weights_class = getattr(py_observers, quant_observer_default)
                        qconfig = observer_weights_class
                    except NotImplementedError:
                        print("Implement your costum observer in pytorch_quantization/observers")

        print(qconfig)

        net.qconfig = qconfig

        if quant_type == "qat":

            torch.quantization.prepare_qat(net, inplace=True)

            
            net = torch.nn.DataParallel(net)
            net.to(device)

            epochs = config["epochs"]

            try: 
                loss_pytorch = importlib.import_module("torch.nn")
                loss_name = config["loss"]["loss"]
                loss_class = getattr(loss_pytorch,loss_name)
                loss_kwargs = config["loss"]["kwargs"]
                criterion = loss_class(**loss_kwargs)
            except NotImplementedError:
                print("Please use a PyTorch valid loss function!")

            try: 
                optim_pytorch = importlib.import_module("torch.optim")
                optim_name = config["optimizer"]["optimizer"]
                optim_class = getattr(optim_pytorch,optim_name)
                optim_kwargs = config["optimizer"]["kwargs"]
                optimizer = optim_class(net.parameters(),**optim_kwargs)
            except NotImplementedError:
                print("Please use a PyTorch valid optimizer!")
            
            
            acc_min = 0.0
            writer = SummaryWriter(log_dir=path_to_tens)
            net.train()
            try: 
                lr_pytorch = importlib.import_module("torch.optim.lr_scheduler")
                lr_scheduler_name = config["lr_scheduler"]["lr_scheduler"]
                lr_scheduler_class = getattr(lr_pytorch,lr_scheduler_name)
                lr_scheduler_kwargs = config["lr_scheduler"]["kwargs"]
                lr_scheduler = lr_scheduler_class(*lr_scheduler_kwargs)
                print("Learning rate scheduler setted up!\n")
            except: 
                pass
            
            metric_name = config["metric"]["metric"]
            try:
                metric = getattr(importlib.import_module("pytorch_quantization.meters"), metric_name)
            except NotImplementedError:
                print("Implement the metric in pytorch-quantization/meters")
            for epoch in range(epochs):  # loop over the dataset multiple times

                val_loss = 0.0
                acc_val = 0.0
                running_loss = 0.0
                acc_train = 0.0
                
                for i, data in enumerate(train_dataloader, 0):
                    batch_loss = 0.0
                    batch_acc = 0.0 
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    if epoch>=3:
                        net.apply(torch.quantization.disable_observer)

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    batch_loss += loss/outputs.shape[0]
                    acc = metric(outputs, labels)
                    batch_acc = acc
                    print('[%d, %5d] ' %
                            (epoch + 1, i + 1), batch_loss, batch_acc)
                    
                    running_loss += loss
                    acc_train += acc

                
                
                net.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        inputs, labels = data

                        
                        inputs = inputs.to(device)  
                        labels = labels.to(device)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        acc = metric(outputs, labels)

                        val_loss += loss
                        acc_val += acc



                try:
                    lr_scheduler.step(val_loss)
                except:
                    pass
                
                print('Val/Loss         :' + str(val_loss/len(val_dataloader))+'   ,' + 'Val/Acc   :' + str(acc_val/len(val_dataloader)))
                if epoch == 0 and "PACT" in quant_type:
                    
                    for el in get_children(net.module):
                        try:
                            el.setup = True
                        except:
                            pass
                if acc_val > acc_min:
                    acc_min = acc_val
                    print('Saving best checkpoints')
                    path = os.path.join(path_save, 'best.pth')
                    torch.save(net.state_dict(),path)
                elif epoch%5==0:
                    path = os.path.join(path_save, 'epoch'+ str(epoch)+ '.pth')
                    torch.save(net.state_dict(), path)
                writer.add_scalar("Loss/train", running_loss.data.item() , epoch)
                writer.add_scalar("Loss/val", val_loss.data.item()/len(val_dataloader) , epoch)
                writer.add_scalar("Acc/train", acc/len(train_dataloader), epoch)
                writer.add_scalar("Acc/val", acc_val/len(val_dataloader) , epoch)
            print('Finished Training')
            
            writer.flush()
            writer.close()


            path_to_load = os.path.join(path_save, 'best.pth')
            net.load_state_dict(torch.load(path_to_load))
            net.to("cpu")
            net.module.to("cpu")
            quantized_model = torch.ao.quantization.convert(net.eval(), inplace=False)
            path = os.path.join(path_save, 'quantized_weights.pth')
            torch.save(quantized_model.state_dict(), path)

        elif quant_type == "static":
            # set validation state
            weights_path = config['quant']["weights_path"]
            try:
                    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
            except:
                chk = adjust_checkpoint(net, weights_path)
                net.load_state_dict(chk)
            net.eval()
            try:
                num_calibration_batches = config["quant"]["num_calibration_batches"]
            except KeyError:

                print("In static mode you need to specify the number of calibration batches (num_calibration_batches)!")

            net.eval()
            try: 
                loss_pytorch = importlib.import_module("torch.nn")
                loss_name = config["loss"]["loss"]
                loss_class = getattr(loss_pytorch,loss_name)
                loss_kwargs = config["loss"]["kwargs"]
                criterion = loss_class(**loss_kwargs)
            except NotImplementedError:
                print("Please use a PyTorch valid loss function!")
            metric_name = config["metric"]["metric"]
            try:
                metric = getattr(importlib.import_module("pytorch_quantization.meters"), metric_name)
            except NotImplementedError:
                print("Implement the metric in pytorch-quantization/meters")
        


            # Prepare the model for static quantization. This inserts observers in
            # the model that will observe activation tensors during calibration.
            torch.quantization.prepare(net, inplace = True)

            # calibrate the prepared model to determine quantization parameters for activations
            # in a real world setting, the calibration would be done with a representative dataset
            # Calibrate with the training set
            current_n = 0
            val_loss = 0.0
            acc_val = 0.0
            with torch.no_grad():
                    for data in val_dataloader:
                        if current_n <= num_calibration_batches:
                            inputs, labels = data

                            
                            inputs = inputs.to(device)  
                            labels = labels.to(device)

                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            acc = metric(outputs, labels)

                            val_loss += loss
                            acc_val += acc
                            current_n += 1
                        else: 
                            break
            

            print('Post Training Quantization: Calibration done')

            #net


            # Convert the observed model to a quantized model. This does several things:
            # quantizes the weights, computes and stores the scale and bias value to be
            # used with each activation tensor, and replaces key operators with quantized
            # implementations.
            torch.quantization.convert(net, inplace = True)
            
            net.eval()
            val_loss = 0.0
            acc_val = 0.0
            with torch.no_grad():
                    for data in val_dataloader:
                    
                        inputs, labels = data

                        
                        inputs = inputs.to(device)  
                        labels = labels.to(device)

                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        acc = metric(outputs, labels)

                        val_loss += loss
                        acc_val += acc
                        current_n += 1

            print('Val/Loss         :' + str(val_loss/len(val_dataloader))+'   ,' + 'Val/Acc   :' + str(acc_val/len(val_dataloader)))
            path = os.path.join(path_save, 'quantized_weights.pth')
            torch.save(net.state_dict(), path)

import argparse

parser = argparse.ArgumentParser(description='Train a network')
parser.add_argument('path_to_experiment',  type=str,
                    help='path to the experiment folder')


args = parser.parse_args()

train(args.path_to_experiment)


