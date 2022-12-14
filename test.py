import torch 

import os
import multiprocessing
import json
import importlib
from torch.quantization.qconfig import QConfig
from experiments.si_cura_float.data import load_data
from pytorch_quantization.utils import adjust_checkpoint

def test_one(data, net, metric):
    with torch.no_grad():
        inputs, labels = data


        outputs = net(inputs)
        acc = metric(outputs, labels)
       
        
        return acc

def test(path_to_experiment, path_to_weights):



    config = os.path.join(path_to_experiment, 'config.json')
    try: 
        with open(config) as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        print("You need to write config.json")
    weights = os.path.join(path_to_experiment, 'logs/saves/best.pth')
    if path_to_weights != '.':
        weights = path_to_weights
    

    net_class = config['model']['model']
    net_kwargs = config['model']['kwargs']
    
    pyquant_models      = importlib.import_module("pytorch_quantization.models")
    try:
        net_class = getattr(pyquant_models, net_class)
        net = net_class(**net_kwargs)
        try:
            net = net.net
        except: 
            pass
    except ModuleNotFoundError:
        print("Implement your model in pytorch_quantization/models")
    train_dataloader, val_dataloader, test_dataloader = load_data()
    if len(load_data()) == 2:
        print("Perform testing on validation set!")
        test_dataloader = val_dataloader


    metric_name = config["metric"]["metric"]
    try:
        metric = getattr(importlib.import_module("pytorch_quantization.meters"), metric_name)
    except NotImplementedError:
        print("Implement the metric in pytorch-quantization/meters")

    try:
        quant = config["quant"]["quant"]
    except: 
        quant = False

    
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
                py_observers = importlib.import_module("torch.quantization")
                observer_act_class = getattr(py_observers, quant_observer_activations)
                observer_weights_class = getattr(py_observers, quant_observer_weights)
               
                if "dtype" in quant_observer_args_activations.keys():
                    quant_observer_args_activations["dtype"] = getattr(torch, quant_observer_args_activations["dtype"])
                if "qscheme" in quant_observer_args_activations.keys():    
                    quant_observer_args_activations["qscheme"] = getattr(torch, quant_observer_args_activations["qscheme"])
                
                observer_weights_class = getattr(py_observers, quant_observer_weights)
                
                if "dtype" in quant_observer_args_weights.keys():
                    quant_observer_args_weights["dtype"] = getattr(torch, quant_observer_args_weights["dtype"])
                if "qscheme" in quant_observer_args_weights.keys():  
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
            # load the real state dict
            try:
                net.load_state_dict(torch.load(weights, map_location='cpu'))
            except:
                chk = adjust_checkpoint(net, weights)
                net.load_state_dict(chk)
            torch.quantization.convert(net, inplace=True)
            
        


                
            net.eval()
            acc_val = 0.0
            import time
            start_time = time.time()

            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data


                    outputs = net(inputs)
                    acc = metric(outputs, labels)

                
                    acc_val += acc
                
                print('Test/Acc   :' + str(acc_val/len(test_dataloader)))
                print("--- %s seconds ---" % (time.time() - start_time))

        elif quant_type == "static":
            # set validation state

            net.eval()

            # Prepare the model for static quantization. This inserts observers in
            # the model that will observe activation tensors during calibration.
            model_fp32_prepared = torch.quantization.prepare(net, inplace = False)
            
            
            torch.quantization.convert(model_fp32_prepared, inplace = True)
            weights = os.path.join(path_to_experiment, 'logs/saves/quantized_weights.pth')
            try:
                model_fp32_prepared.load_state_dict(torch.load(weights, map_location='cpu'))
            except:
                chk = adjust_checkpoint(model_fp32_prepared, weights)
                model_fp32_prepared.load_state_dict(chk)

            acc_val = 0.0
            model_fp32_prepared.eval()
            import time
            start_time = time.time()

            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data


                    outputs = model_fp32_prepared(inputs)
                    acc = metric(outputs, labels)

                
                    acc_val += acc
                
                print('Test/Acc   :' + str(acc_val/len(test_dataloader)))
                print("--- %s seconds ---" % (time.time() - start_time))
                print(quant_type)

        elif quant_type == "dynamic":

            # set validation state
            weights_path = config['quant']["weights_path"]
            dynamic_args = config["quant"]["dyn"]

            dynamic_args["dtype"] = getattr(torch, dynamic_args["dtype"])
            substitute = {}
            for el in dynamic_args["layers"]:
                add = getattr(torch.nn, el)
                substitute.add(add)

            dynamic_args["layers"] = substitute

            try:
                    net.load_state_dict(torch.load(weights_path, map_location='cpu'))
            except:
                chk = adjust_checkpoint(net, weights_path)
                net.load_state_dict(chk)
            net.eval()

            net.eval()
            metric_name = config["metric"]["metric"]
            try:
                metric = getattr(importlib.import_module("pytorch_quantization.meters"), metric_name)
            except NotImplementedError:
                print("Implement the metric in pytorch-quantization/meters")
    



            torch.quantization.quantize_dynamic(model=net, **dynamic_args)
            
            net.eval()
            val_loss = 0.0
            acc_val = 0.0
            with torch.no_grad():
                    for data in val_dataloader:
                    
                        inputs, labels = data

                        outputs = net(inputs)
                        acc = metric(outputs, labels)

                        acc_val += acc

            print('Test/Loss         :' + str(val_loss/len(val_dataloader))+'   ,' + 'Test/Acc   :' + str(acc_val/len(val_dataloader)))
        
    else:
        try:
            net.load_state_dict(torch.load(weights, map_location='cpu'))
        except:
            chk = adjust_checkpoint(net, weights)
            net.load_state_dict(chk)

        net.eval()
        acc_val = 0.0
        import time
        start_time = time.time()
        try:
            n_cpus = config["n_cpus"]
        except:
            n_cpus = multiprocessing.cpu_count()
            
        pool = multiprocessing.Pool(n_cpus)
        net.eval()
        data_list = []
        
        for data in test_dataloader: 
            data_list.append((data,net, metric))
        
        results = pool.starmap(test_one, data_list)   
    

        acc_val = sum(results)
        print('Test/Acc   :' + str(acc_val/len(results)))
        print("--- %s seconds ---" % (time.time() - start_time))


import argparse

parser = argparse.ArgumentParser(description='Train a network')
parser.add_argument('path_to_experiment',  type=str,
                    help='path to the experiment folder')
parser.add_argument('-path_to_weights',  type=str, default='.',
                    help='path to checkpoint')



args = parser.parse_args()
if __name__ == '__main__':
    test(args.path_to_experiment, args.path_to_weights)