import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
from data_list import ImageList
from torch.autograd import Variable
import torch.nn.functional as F
from nltk.tbl import feature
from random import shuffle
import time
from pre_process_DW_Reuters_Caltec import load_officecaltec_dataset,load_reuters_dataset_random
optim_dict = {"SGD": optim.SGD, "ADAM":optim.Adam, "RMSprop":optim.RMSprop, "AdaGrad":optim.Adagrad}
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
source_lang='FR'

random.seed(1364)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.fill_(0.0)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def rand_choose_10percat_target(target_X,target_y):
    ### Randomly choose 10 from each category as training
    random.seed(11) 
    sample_size = 10
    # append target_X and target_y
    target_X_train_sampled = []
    target_y_train_sampled = []
    target_X_test_sampled = []
    target_y_test_sampled = []
    for i in range(6):
        target_X_i = target_X[np.where(target_y == i)[0]]
        idx_train = random.sample(range(0,len(target_X_i)), sample_size)

        idx_test = list(set(np.arange(0, len(target_X_i))) - set(idx_train))

        target_X_train_i_sampled = target_X_i[idx_train]
        target_y_train_i_sampled = [i] * sample_size
        for e in target_X_train_i_sampled:
            target_X_train_sampled.append(e)
        for e in target_y_train_i_sampled:
            target_y_train_sampled.append(e)
        
        target_X_test_i_sampled = target_X_i[idx_test]
        target_y_test_i_sampled = [i] * (len(target_X_i) - sample_size)
        for e in target_X_test_i_sampled:
            target_X_test_sampled.append(e)
        for e in target_y_test_i_sampled:
            target_y_test_sampled.append(e)
            
    target_X_train_sampled = np.vstack(target_X_train_sampled)
    target_X_train_sampled = np.array(target_X_train_sampled)
    target_y_train_sampled = np.vstack(target_y_train_sampled)
    target_y_train_sampled = np.array(target_y_train_sampled)
    
    target_X_test_sampled = np.vstack(target_X_test_sampled)
    target_X_test_sampled = np.array(target_X_test_sampled)
    target_y_test_sampled = np.vstack(target_y_test_sampled)
    target_y_test_sampled = np.array(target_y_test_sampled)
    
    
    return target_X_train_sampled, target_y_train_sampled, target_X_test_sampled, target_y_test_sampled

#################### BaseNetworks without Feature Alignment##########################
#** -> SP
target_X,target_y=load_reuters_dataset_random(lang='SP', dim=0)
source_X,source_y=load_reuters_dataset_random(lang=source_lang, dim=0)
target_y=target_y.reshape(-1,1)
source_y=source_y.reshape(-1,1)

N=len(source_X)

print(N)
s = np.arange(N)
#np.random.shuffle(s)
source_X,source_y=source_X[s],source_y[s]

_,source_dim=source_X.shape
_,target_dim=target_X.shape

print(source_y.shape)
print(source_X.shape)
print(target_X.shape)

### Randomly choose 3 from each category as training
target_X_train, target_y_train, target_X_test, target_y_test = rand_choose_10percat_target(target_X,target_y)

# This is used in target classifier in semisupervised mode
target_X_train = torch.tensor(target_X_train,dtype=torch.float)
target_y_train = torch.tensor(target_y_train,dtype=torch.long) # test data is converted to tensor in image_classification test function 


## shuffle train
N=len(target_X_train)
s = np.arange(N)
np.random.shuffle(s)
target_X_train,target_y_train=target_X_train[s],target_y_train[s]

## shuffle test
N=len(target_X_test)
s = np.arange(N)
np.random.shuffle(s)

target_X_test,target_y_test=target_X_test[s],target_y_test[s]

N=len(target_X)
s = np.arange(N)
np.random.shuffle(s)

target_X=target_X[s]

eyeData = torch.eye(200).to('cuda')

def visual_Data_t(model, gpu=True):

    inputs = torch.tensor(target_X_test,dtype=torch.float)
    labels=torch.tensor(target_y_test,dtype=torch.long)
    if gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)
    features_aligned = model(inputs)

def visual_Data_s(model, gpu=True):

    inputs = torch.tensor(source_X,dtype=torch.float)
    labels=torch.tensor(source_y,dtype=torch.long)

    if gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)
    features_aligned = model(inputs)
    
def text_classification_test(model, gpu=True):
    start_test = True
    target_len=target_y_test.size

    inputs = torch.tensor(target_X_test,dtype=torch.float)
    labels=torch.tensor(target_y_test,dtype=torch.long)

    if gpu:
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
    else:
        inputs = Variable(inputs)
        labels = Variable(labels)
    outputs = model(inputs)
   
    _, predict = torch.max(outputs, 1)

    validate_set=predict.reshape(target_len,1)[:int(target_len/2)]
    test_set=predict.reshape(target_len,1)[int(target_len/2):]
    labels_val=labels[:int(target_len/2)]
    labels_test=labels[int(target_len/2):]
    

    trueNumber=float(torch.sum(predict.reshape(target_len,1) == labels))
    totalNumber=float(labels.size()[0])

    accuracy = torch.div(trueNumber, totalNumber)

    trueNumber=float(torch.sum(validate_set == labels_val))
    totalNumber=float(labels_val.size()[0])
    accuracy_val = torch.div(trueNumber, totalNumber)

    trueNumber=float(torch.sum(test_set == labels_test))
    totalNumber=float(labels_test.size()[0])
    accuracy_test = torch.div(trueNumber, totalNumber)


    return accuracy.item(),accuracy_val.item(),accuracy_test.item()

def rec_loss_cal(rec_samples, inputs):
    return torch.dist(rec_samples, inputs, p=2) 

number=time.time()


class MyClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter variables
        if hasattr(module, 'weight'):
            w = module.weight.data
            norms = torch.sqrt(torch.sum(torch.mul(w,w), dim=1, keepdim=True))
            desired = torch.clamp(input=norms,max=100)
            w.div_(torch.div(norms,desired).expand_as(w))

clipper = MyClipper()

def transfer_classification(config):
    class_criterion = nn.MultiMarginLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}
    class_num = 6
    ## set base networks
    net_config = config["network"]
    base_network_s = network.network_dict[net_config["name_s"]]()
    base_network_t = network.network_dict[net_config["name_t"]]()
    P_network_s = network.network_dict[net_config["name_s"]]()
    P_network_t = network.network_dict[net_config["name_t"]]()
    base_space_reverse=network.network_dict["base_space"]()
    generator_mmd=network.network_dict["generator_mmd"]()
    discriminator_mmd=network.network_dict["MMD_discriminator"]()
    classifier_layer_t=nn.Sequential(
        nn.Linear(generator_mmd.output_num(), class_num),
        )
    classifier_layer_s=nn.Sequential(
        nn.Linear(generator_mmd.output_num(), class_num),
        )

    reconstruct_common=nn.Sequential(
        nn.Linear(base_space_reverse.output_num(), base_space_reverse.output_num(),bias=False),
        )

    reconstruct_s=nn.Sequential(
        nn.Linear(base_space_reverse.output_num(), source_dim,bias=False),
        )
    reconstruct_t=nn.Sequential(
        nn.Linear(base_space_reverse.output_num(), target_dim,bias=False),
        )

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer_t = classifier_layer_t.cuda()
        classifier_layer_s = classifier_layer_s.cuda()
        base_space_reverse=base_space_reverse.cuda()
        discriminator_mmd=discriminator_mmd.cuda()
        generator_mmd=generator_mmd.cuda()
        base_network_t = base_network_t.cuda()
        base_network_s = base_network_s.cuda()
        P_network_s=P_network_s.cuda()
        P_network_t=P_network_t.cuda()
        reconstruct_s=reconstruct_s.cuda()
        reconstruct_t=reconstruct_t.cuda()
        reconstruct_common=reconstruct_common.cuda()

    ## collect parameters
    parameter_list = [{"params":classifier_layer_s.parameters(), "lr":1},{"params":classifier_layer_t.parameters(), "lr":1},
                      {"params":base_network_s.parameters(), "lr":1}, {"params":base_network_t.parameters(), "lr":1},
                      {"params":base_space_reverse.parameters(), "lr":1},{"params":reconstruct_s.parameters(), "lr":1},
                      {"params":reconstruct_t.parameters(), "lr":1},{"params":reconstruct_common.parameters(), "lr":1},
                      {"params":P_network_s.parameters(), "lr":1},{"params":P_network_t.parameters(), "lr":1}]
    parameter_mmd_list = [{"params":discriminator_mmd.parameters(), "lr":1}]
    parameter_mmd_gen_list = [{"params":generator_mmd.parameters(), "lr":1}]

    assert base_network_s.output_num() == base_network_t.output_num()
    
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    optimizer_mmd = optim_dict[optimizer_config["type"]](parameter_mmd_list, **(optimizer_config["optim_params"]))
    optimizer_mmd_gen = optim_dict[optimizer_config["type"]](parameter_mmd_gen_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    param_lr_mmd = []
    for param_group in optimizer_mmd.param_groups:
        param_lr_mmd.append(param_group["lr"])
    param_lr_mmd_gen = []
    for param_group in optimizer_mmd_gen.param_groups:
        param_lr_mmd_gen.append(param_group["lr"])

    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    ## train   
    transfer_loss = classifier_loss_t = classifier_loss_s = 0
    acc_list=[]

    for epoch in range(config["num_iterations"]):
        ## test in the train
        if epoch % config["test_interval"] == 0:
            classifier_layer_t.train(False)
            classifier_layer_s.train(False)
            base_space_reverse.train(False)
            discriminator_mmd.train(False)
            generator_mmd.train(False)
            base_network_t.train(False)
            base_network_s.train(False)
            P_network_s.train(False)
            P_network_t.train(False)
            reconstruct_s.train(False)
            reconstruct_t.train(False)
            reconstruct_common.train(False)
	    # For visualization purpose
            #visual_Data_t(nn.Sequential(base_network_t, base_space_reverse,generator_mmd), gpu=use_gpu)
            #visual_Data_s(nn.Sequential(base_network_s, base_space_reverse,generator_mmd), gpu=use_gpu)

            acc,valid_acc,test_acc=text_classification_test(nn.Sequential(base_network_t, base_space_reverse,generator_mmd,classifier_layer_t), gpu=use_gpu)
            with open('results/1_batch30_%s_10_0.001_0.003_%s.txt'%(source_lang,str(number)),'a+') as f:
                f.write(str(acc))
                f.write('\n')
            acc_list.append([acc])
            print(acc)
        for i in range(config["n_batches"]):
            
            ## train one iter
            optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
            optimizer_mmd = lr_scheduler(param_lr_mmd, optimizer_mmd, i, **schedule_param)
            optimizer_mmd_gen = lr_scheduler(param_lr_mmd_gen, optimizer_mmd_gen, i, **schedule_param)
            optimizer.zero_grad()
            optimizer_mmd.zero_grad()
            optimizer_mmd_gen.zero_grad()

            target_len=target_y.size
            source_len=source_y.size
            n_batches = config["n_batches"]
    
            local_Xs, local_ys = source_X[i*n_batches:(i+1)*n_batches,], source_y[i*n_batches:(i+1)*n_batches,]
            local_Xt, local_yt = target_X[i*n_batches:(i+1)*n_batches,], target_y[i*n_batches:(i+1)*n_batches,]
            
            local_Xs, local_ys = unison_shuffled_copies(local_Xs, local_ys)
            local_Xt, local_yt = unison_shuffled_copies(local_Xt, local_yt)
            
            if len(local_Xt) < n_batches:
                needed = n_batches - len(local_Xt)
                list_loc_t = [1]*needed + [0]*(len(target_X)- needed)
                shuffle(list_loc_t)
                filters = [x==1 for x in list_loc_t]

                new_needed_samples_t = target_X[filters]
                new_needed_labels_t =target_y[filters]
                local_Xt = np.concatenate((new_needed_samples_t,local_Xt),axis=0)
                local_yt = np.concatenate((new_needed_labels_t,local_yt),axis=0)
        
            if len(local_Xs) < n_batches:
                needed = n_batches - len(local_Xs)
                list_loc_s = [1]*needed + [0]*(len(source_X)- needed)
                shuffle(list_loc_s)
                filters = [x==1 for x in list_loc_s]
                new_needed_samples_s = source_X[filters]
                new_needed_labels_s = source_y[filters]
                local_Xs = np.concatenate((new_needed_samples_s,local_Xs),axis=0)
                local_ys = np.concatenate((new_needed_labels_s,local_ys),axis=0)


            local_Xs = torch.tensor(local_Xs,dtype=torch.float)
            local_ys = torch.tensor(local_ys,dtype=torch.long)
            local_Xt = torch.tensor(local_Xt,dtype=torch.float)
            local_yt = torch.tensor(local_yt,dtype=torch.long)
            
            content_target_tensor=torch.tensor(target_X,dtype=torch.float)
            if use_gpu:
                inputs_source, labels_source, inputs_target, labels_target,content_target = Variable(local_Xs).cuda(), Variable(local_ys).cuda(), Variable(target_X_train).cuda(), Variable(target_y_train).cuda(),Variable(content_target_tensor).cuda()
            else:
                inputs_source, labels_source, inputs_target, labels_target = Variable(local_Xs), Variable(local_ys), Variable(target_X_train), Variable(target_y_train)

            features_s = base_network_s(inputs_source)
            feature_s_basespace=base_space_reverse(features_s)
            aligned_features_s=generator_mmd(feature_s_basespace)
            outputs_s = classifier_layer_t(aligned_features_s)
            classifier_loss_s = class_criterion(outputs_s, labels_source.reshape(n_batches,))

            features_t = base_network_t(inputs_target)
            feature_t_basespace=base_space_reverse(features_t)

            prejected_source=P_network_s(inputs_source)
            prejected_target=P_network_t(inputs_target)

            reconstructed_s=reconstruct_common(feature_s_basespace)
            reconstructed_t=reconstruct_common(feature_t_basespace)

            reconstruct_loss_s=rec_loss_cal(reconstructed_s,prejected_source)
            reconstruct_loss_t=rec_loss_cal(reconstructed_t,prejected_target)
            
            l1_regularization = torch.norm(feature_s_basespace, 1)+ torch.norm(feature_t_basespace, 1)

            aligned_features_t=generator_mmd(feature_t_basespace)

            features_t_c = base_network_t(content_target)
            feature_t_c_basespace=base_space_reverse(features_t_c)
            aligned_features_c_t=generator_mmd(feature_t_c_basespace)

            outputs_t = classifier_layer_t(aligned_features_t)

            classifier_loss_t = class_criterion(outputs_t, labels_target.reshape(len(labels_target),))
            
            
            feature_s_mmd=discriminator_mmd(aligned_features_s)
            feature_t_mmd=discriminator_mmd(aligned_features_c_t)
            transfer_loss = transfer_criterion(feature_s_mmd, feature_t_mmd, **loss_config["params"])
            
            
            for w_i in base_space_reverse.parameters():
                A_F_2=torch.norm(torch.transpose(w_i,0,1),2)
               
            for w_i in base_network_s.parameters():
                temp_value = (torch.norm((torch.mm(w_i,torch.transpose(w_i, 0, 1))-eyeData),2))
                B_s_reg=torch.mul(temp_value,temp_value)
                
            for w_i in base_network_t.parameters():
                temp_value = (torch.norm((torch.mm(w_i,torch.transpose(w_i, 0, 1))-eyeData),2))
                B_t_reg=torch.mul(temp_value,temp_value)

            for w_i in reconstruct_s.parameters():
                temp_value = (torch.norm((torch.mm(torch.transpose(w_i, 0, 1),w_i)-eyeData),2))
                P_s_reg=torch.mul(temp_value,temp_value)
                P_s_F_2=torch.norm(torch.transpose(w_i,0,1),2)

            for w_i in reconstruct_t.parameters():
                temp_value = (torch.norm((torch.mm(torch.transpose(w_i, 0, 1),w_i)-eyeData),2))
                P_t_reg=torch.mul(temp_value,temp_value)
                P_t_F_2=torch.norm(torch.transpose(w_i,0,1),2)

            for w_i in P_network_s.parameters():
                temp_value = (torch.norm((torch.mm(w_i,torch.transpose(w_i, 0, 1))-eyeData),2))
                Project_s_reg=torch.mul(temp_value,temp_value)

            for w_i in P_network_t.parameters():
                temp_value = (torch.norm((torch.mm(w_i,torch.transpose(w_i, 0, 1))-eyeData),2))
                Project_t_reg=torch.mul(temp_value,temp_value)
            for w_i in reconstruct_common.parameters():
                D_F_2=torch.norm(torch.transpose(w_i,0,1),2)


            classifier_layer_t.train(True)
            classifier_layer_s.train(True)
            base_space_reverse.train(True)
            discriminator_mmd.train(False)
            generator_mmd.train(True)
            base_network_t.train(True)
            base_network_s.train(True)
            reconstruct_s.train(True)
            reconstruct_t.train(True)
            reconstruct_common.train(True)
	    
	    coef = 1e-8
            total_loss = 10*classifier_loss_t+ classifier_loss_s+ coef*reconstruct_loss_t\
                +coef*reconstruct_loss_s+coef*A_F_2+coef*Project_s_reg+coef*Project_t_reg\
                +coef*D_F_2+coef*B_s_reg+coef*B_t_reg

            total_loss.backward(retain_graph=True)
            optimizer.step()
            
            base_space_reverse.apply(clipper)
          
            classifier_layer_t.train(False)
            classifier_layer_s.train(False)
            base_space_reverse.train(False)
            discriminator_mmd.train(True)
            generator_mmd.train(False)
            base_network_t.train(False)
            base_network_s.train(False)
            reconstruct_s.train(False)
            reconstruct_t.train(False)
            reconstruct_common.train(True)
            transfer_loss_reverse=-transfer_loss
            transfer_loss_reverse.backward(retain_graph=True)
            optimizer_mmd.step()

            classifier_layer_t.train(False)
            classifier_layer_s.train(False)
            base_space_reverse.train(False)
            discriminator_mmd.train(False)
            generator_mmd.train(True)
            base_network_t.train(False)
            base_network_s.train(False)
            reconstruct_s.train(False)
            reconstruct_t.train(False)
            reconstruct_common.train(False)
            transfer_loss_=transfer_loss
            transfer_loss_.backward()
            optimizer_mmd_gen.step()


if __name__ == "__main__":

    config = {}
    config["num_iterations"] = 3000
    config["test_interval"] = 10
    config["n_batches"] = 30
    config["loss"] = {"name":'DAN', "trade_off":1.0 }
    #config["data"] = [{"name":"source", "type":"image", "list_path":{"train":"./data/office-full/"+args.source+"_list.txt"}, "batch_size":{"train":36, "test":4} }, {"name":"target", "type":"image", "list_path":{"train":"./data/office-full/"+args.target+"_list.txt"}, "batch_size":{"train":36, "test":4} }]
    config["network"] = {"name_s":source_lang,"name_t":"SP","use_bottleneck":0, "bottleneck_dim":256}
    #config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.0003, "gamma":0.0003, "power":0.75} }
    #config["optimizer"] = {"type":"ADAM", "optim_params":{"lr":0.0001, "eps":1e-8, "weight_decay":0.01}, "lr_type":"inv", "lr_param":{"init_lr":0.0003, "gamma":1, "power":0.75}}
    config["optimizer"] = {"type":"ADAM", "optim_params":{"lr":0.0001, "eps":1e-7, "weight_decay":0.001}, "lr_type":"inv", "lr_param":{"init_lr":0.0005, "gamma":1, "power":0.75}}

    #config["optimizer"] = {"type":"AdaGrad", "optim_params":{"lr":0.1}}
    #print config["loss"]
    transfer_classification(config)
    
