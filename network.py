import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

import torch.nn.functional as F

dimension_latent=1024
dimension_latent_text= 200 
MMD_dim=200
MMD_dim_out=100

#######################
#### image encoders ###

class encoder_surf(nn.Module):
    def __init__(self):
        super(encoder_surf, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(800, dimension_latent),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class surf(nn.Module):
    def __init__(self):
        super(surf, self).__init__()
        model = encoder_surf()
        self.features = model.features
        self.__in_features = dimension_latent
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent)
        return x

    def output_num(self):
        return self.__in_features

class decoder_surf(nn.Module):
    def __init__(self):
        super(decoder_surf, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 800),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 800)
        return x

class encoder_decaf(nn.Module):
    def __init__(self):
        super(encoder_decaf, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(4096, dimension_latent),
        )

    def forward(self, x):
        x = self.features(x)
        return x

class decaf(nn.Module):
    def __init__(self):
        super(decaf, self).__init__()
        model_alexnet = encoder_decaf()
        self.features = model_alexnet.features
        self.__in_features =  dimension_latent# last max pooling layer
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent)
        return x

    def output_num(self):
        return self.__in_features

class decoder_decaf(nn.Module):
    def __init__(self):
        super(decoder_decaf, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 4096),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 4096)
        return x

class Decoder_ConvNet(nn.Module):
    #256 * 6 * 6 --> 3 *224 * 224
    def __init__(self):
        super(Decoder_ConvNet, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=6, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=9, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=31, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, kernel_size=10, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), 256, 6, 6)
        x = self.ConvLayers(x)
        #print x.shape
        x = x.view(x.size(0), 3 * 224 * 224)
        return x

#######################
#### text encoders ####

# SP
class encoder_SP(nn.Module):
    def __init__(self):
        super(encoder_SP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(11546, dimension_latent_text,bias=False),
        )
        #self.features.apply(init_weights)
    def forward(self, x):
        x = self.features(x)
        return x     

class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        model = encoder_SP()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features

class decoder_SP(nn.Module):
    def __init__(self):
        super(decoder_SP, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent_text, 100),
            nn.Linear(100, 11546),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 11546)
        return x

# EN 
class encoder_EN(nn.Module):
    def __init__(self):
        super(encoder_EN, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(21529, dimension_latent_text,bias=False),
        )
        #self.features.apply(init_weights)
    def forward(self, x):
        x = self.features(x)
        return x

class EN(nn.Module):
    def __init__(self):
        super(EN, self).__init__()
        model = encoder_EN()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features

class decoder_EN(nn.Module):
    def __init__(self):
        super(decoder_EN, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent_text, 100),
            nn.Linear(100, 21531),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 21531)
        return x
    
# GR
class encoder_GR(nn.Module):
    def __init__(self):
        super(encoder_GR, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(34269, dimension_latent_text,bias=False),
        )
    def forward(self, x):
        x = self.features(x)
        return x  

class GR(nn.Module):
    def __init__(self):
        super(GR, self).__init__()
        model = encoder_GR()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features

class decoder_GR(nn.Module):
    def __init__(self):
        super(decoder_GR, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent_text, 100),
            nn.Linear(100, 34268),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 34268)
        return x

# IT
class encoder_IT(nn.Module):
    def __init__(self):
        super(encoder_IT, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(15499, dimension_latent_text,bias=False)
        )
    def forward(self, x):
        x = self.features(x)
        return x

class IT(nn.Module):
    def __init__(self):
        super(IT, self).__init__()
        model = encoder_IT()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features

class decoder_IT(nn.Module):
    def __init__(self):
        super(decoder_IT, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent_text, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 15499),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 15499)
        return x 

# FR
class encoder_FR(nn.Module):
    def __init__(self):
        super(encoder_FR, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(24884, dimension_latent_text,bias=False),
        )
    def forward(self, x):
        x = self.features(x)
        return x
  
class FR(nn.Module):
    def __init__(self):
        super(FR, self).__init__()
        model = encoder_FR()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features

class decoder_FR(nn.Module):
    def __init__(self):
        super(decoder_FR, self).__init__()
        self.ConvLayers = nn.Sequential(
            nn.Linear(dimension_latent_text, 100),
            nn.Linear(100, 24894),
        )
    
    def forward(self, x):
        x = self.ConvLayers(x)
        x = x.view(x.size(0), 24894)
        return x

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
    
# base_space
class model_base_space(nn.Module):
    def __init__(self):
        super(model_base_space, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(dimension_latent_text, dimension_latent_text,bias=False),
        )
        self.features.apply(init_weights)


    def forward(self, x):
        x = self.features(x)
        return x
  
class base_space(nn.Module):
    def __init__(self):
        super(base_space, self).__init__()
        model = model_base_space()
        self.features = model.features
        self.__in_features = dimension_latent_text
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), dimension_latent_text)
        return x

    def output_num(self):
        return self.__in_features
      
# MMD
class model_MMD_discriminator(nn.Module):
    def __init__(self):
        super(model_MMD_discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(MMD_dim, MMD_dim_out),
        )

    def forward(self, x):
        x = self.features(x)
        return x
  
class MMD_discriminator(nn.Module):
    def __init__(self):
        super(MMD_discriminator, self).__init__()
        model = model_MMD_discriminator()
        self.features = model.features
        self.__in_features = MMD_dim_out
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), MMD_dim_out)
        return x

    def output_num(self):
        return self.__in_features       

# MMD_encoder
class model_generator_MMD(nn.Module):
    def __init__(self):
        super(model_generator_MMD, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(dimension_latent_text, MMD_dim),
        )

    def forward(self, x):
        x = self.features(x)
        return x
  
class generator_mmd(nn.Module):
    def __init__(self):
        super(generator_mmd, self).__init__()
        model = model_generator_MMD()
        self.features = model.features
        self.__in_features = MMD_dim
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), MMD_dim)
        return x

    def output_num(self):
        return self.__in_features 

# convNet without the last layer
class AlexNetFc_s(nn.Module):
  def __init__(self):
    super(AlexNetFc_s, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    # No need to add classifier layers of alexNet: Just need the feature extractors
    #   self.classifier = nn.Sequential()
    #   for i in xrange(6):
    #       self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    #  self.__in_features = model_alexnet.classifier[6].in_features
    self.__in_features =  256 * 6 * 6# last max pooling layer
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    return x

  def output_num(self):
    return self.__in_features


class AlexNetFc_t(nn.Module):
  def __init__(self):
    super(AlexNetFc_t, self).__init__()
    model_alexnet_t = models.alexnet(pretrained=True)
    self.features = model_alexnet_t.features
    self.__in_features = 256 * 6 * 6 # last max pooling layer
    
  def forward(self, x_t):
    x_t = self.features(x_t)
    x_t = x_t.view(x_t.size(0), 256*6*6)
    #?
    #x_t = self.classifier(x_t)
    return x_t

  def output_num(self):
    return self.__in_features
        

network_dict = {"surf":surf,"decaf":decaf,"SP":SP,"EN":EN,"FR":FR,"GR":GR,"IT":IT,"base_space":base_space,"MMD_discriminator":MMD_discriminator,
                "generator_mmd":generator_mmd,"AlexNet_s":AlexNetFc_s, "AlexNet_t":AlexNetFc_t, "ResNet18":ResNet18Fc, "ResNet34":ResNet34Fc, "ResNet50":ResNet50Fc, "ResNet101":ResNet101Fc, "ResNet152":ResNet152Fc}
