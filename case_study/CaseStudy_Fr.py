import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import network_Casestudy as network
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
from pre_process_DW_Reuters_Caltec import load_reuters_dataset_random_10labels
optim_dict = {"SGD": optim.SGD, "ADAM":optim.Adam, "RMSprop":optim.RMSprop, "AdaGrad":optim.Adagrad}

source_lang='EN'

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
#target_X,target_y=load_reuters_dataset_random_10labels(lang='SP', dim=0)
#source_X,source_y=load_reuters_dataset_random_10labels(lang=source_lang, dim=0)
print "loading data ..."

source_X = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_En_train.txt")
source_y = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_En_Y_train.txt")

source_y=source_y.reshape(-1,1)

print "En loaded!"
                                                                                    
target_X_train = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_Fr_train.txt")
target_y_train = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_Fr_Y_train.txt")

print "Fr train loaded!"

target_X_test = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_Fr_test.txt")
target_y_test = np.loadtxt("/CaseStudy_DA/Vectorized/casestudy_DA_Fr_Y_test.txt")

target_y_train=target_y_train.reshape(-1,1)
target_y_test=target_y_test.reshape(-1,1)

print "Fr test loaded!"

target_X = np.vstack((target_X_train, target_X_test))
target_y = np.vstack((target_y_train, target_y_test))


_,source_dim=source_X.shape
_,target_dim=target_X_train.shape
# 
print(source_y.shape)
print(source_X.shape)
print(target_X_train.shape)


### Randomly choose 3 from each category as training
#target_X_train, target_y_train, target_X_test, target_y_test = rand_choose_10percat_target(target_X,target_y)

# This is used in target classifier when labeled data is available
target_X_train = torch.tensor(target_X_train,dtype=torch.float)
target_y_train = torch.tensor(target_y_train,dtype=torch.long) # test data is converted to tensor in image_classification test function 


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
    #np.savetxt('target_label.txt',labels.cpu().data.numpy()+6,fmt='%d')
    #np.savetxt('target_inputs.txt',inputs.cpu().data.numpy())
    #np.savetxt('target_feature.txt',features_aligned.cpu().data.numpy())

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
    outputs = torch.nn.functional.softmax(outputs)
    np.savetxt('casestudy_predict_Fr.txt', outputs.cpu().data,fmt='%1.4f')
   
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
    return torch.norm(rec_samples - inputs) 

number=time.time()

def transfer_classification(config):

    class_criterion = nn.MultiMarginLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}
    class_num = 2
    ## set base networks
    net_config = config["network"]
    base_network_s = network.network_dict[net_config["name_s"]]()
    base_network_t = network.network_dict[net_config["name_t"]]()
    base_space_reverse=network.network_dict["base_space"]() #New
    generator_mmd=network.network_dict["generator_mmd"]()
    discriminator_mmd=network.network_dict["MMD_discriminator"]() #New
    classifier_layer_t=nn.Sequential(
        nn.Linear(generator_mmd.output_num(), class_num),
        )
    classifier_layer_s=nn.Sequential(
        nn.Linear(generator_mmd.output_num(), class_num),
        )
    reconstruct_s=nn.Sequential(
        nn.Linear(base_space_reverse.output_num(), source_dim),
        )
    reconstruct_t=nn.Sequential(
        nn.Linear(base_space_reverse.output_num(), target_dim),
        )
    
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        classifier_layer_t = classifier_layer_t.cuda()
        classifier_layer_s = classifier_layer_s.cuda()
        base_space_reverse=base_space_reverse.cuda() #New
        discriminator_mmd=discriminator_mmd.cuda()
        generator_mmd=generator_mmd.cuda()
        base_network_t = base_network_t.cuda()
        base_network_s = base_network_s.cuda()
        reconstruct_s=reconstruct_s.cuda()
        reconstruct_t=reconstruct_t.cuda()

    ## collect parameters
    parameter_list = [{"params":classifier_layer_s.parameters(), "lr":1},{"params":classifier_layer_t.parameters(), "lr":1},
                      {"params":base_network_s.parameters(), "lr":1}, {"params":base_network_t.parameters(), "lr":1},
                      {"params":base_space_reverse.parameters(), "lr":1},{"params":reconstruct_s.parameters(), "lr":1},
                      {"params":reconstruct_t.parameters(), "lr":1}]
    parameter_mmd_list = [{"params":discriminator_mmd.parameters(), "lr":1}]
    parameter_mmd_gen_list = [{"params":generator_mmd.parameters(), "lr":1}]

    #parameter_list = list(classifier_layer_s.parameters()) + list(classifier_layer_t.parameters()) + list(base_network_s.parameters())+ list(base_network_t.parameters())
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
        #print "epoch %s from %s" % (epoch, config["num_iterations"])  
        if epoch % config["test_interval"] == 0:
            classifier_layer_t.train(False)
            classifier_layer_s.train(False)
            base_space_reverse.train(False)
            discriminator_mmd.train(False)
            generator_mmd.train(False)
            base_network_t.train(False)
            base_network_s.train(False)
            reconstruct_s.train(False)
            reconstruct_t.train(False)
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

            # Only sampled train data from target is fed to the classifier.

            features_t = base_network_t(inputs_target)
            feature_t_basespace=base_space_reverse(features_t)

            reconstructed_s=reconstruct_s(feature_s_basespace)
            reconstructed_t=reconstruct_t(feature_t_basespace)

            reconstruct_loss=rec_loss_cal(reconstructed_s,inputs_source)+rec_loss_cal(reconstructed_t,inputs_target)
            
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
                      

            classifier_layer_t.train(True)
            classifier_layer_s.train(True)
            base_space_reverse.train(True)
            discriminator_mmd.train(False)
            generator_mmd.train(True)
            base_network_t.train(True)
            base_network_s.train(True)
            reconstruct_s.train(True)
            reconstruct_t.train(True)
            total_loss = 10*classifier_loss_t+ classifier_loss_s+0.000001*l1_regularization+0.000001*reconstruct_loss
            total_loss.backward(retain_graph=True)
            optimizer.step()

            '''
            print(classifier_loss_s)
            print(classifier_loss_t)
            print(transfer_loss)
            '''
            #print(transfer_loss)
            
            classifier_layer_t.train(False)
            classifier_layer_s.train(False)
            base_space_reverse.train(False)
            discriminator_mmd.train(True)
            generator_mmd.train(False)
            base_network_t.train(False)
            base_network_s.train(False)
            reconstruct_s.train(False)
            reconstruct_t.train(False)
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
            transfer_loss_=transfer_loss
            transfer_loss_.backward()
            optimizer_mmd_gen.step()

if __name__ == "__main__":

    config = {}
    config["num_iterations"] = 1000
    config["test_interval"] = 10
    config["n_batches"] = 30#20
    config["loss"] = {"name":'DAN', "trade_off":1.0 }
    #config["data"] = [{"name":"source", "type":"image", "list_path":{"train":"./data/office-full/"+args.source+"_list.txt"}, "batch_size":{"train":36, "test":4} }, {"name":"target", "type":"image", "list_path":{"train":"./data/office-full/"+args.target+"_list.txt"}, "batch_size":{"train":36, "test":4} }]
    config["network"] = {"name_s":source_lang,"name_t":"SP","use_bottleneck":0, "bottleneck_dim":256}
    #config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", "lr_param":{"init_lr":0.0003, "gamma":0.0003, "power":0.75} }
    #config["optimizer"] = {"type":"ADAM", "optim_params":{"lr":0.0001, "eps":1e-8, "weight_decay":0.01}, "lr_type":"inv", "lr_param":{"init_lr":0.0003, "gamma":1, "power":0.75}}
    config["optimizer"] = {"type":"ADAM", "optim_params":{"lr":0.0001, "eps":1e-7, "weight_decay":0.001}, "lr_type":"inv", "lr_param":{"init_lr":0.005, "gamma":1, "power":0.75}}

    #config["optimizer"] = {"type":"AdaGrad", "optim_params":{"lr":0.1}}
    #print config["loss"]
    transfer_classification(config)
    
