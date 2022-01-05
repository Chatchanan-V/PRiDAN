# The original creator of the following code is from https://github.com/layumi/Person_reID_baseline_pytorch

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from torchsummary import summary

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=False, num_bottleneck=512, linear=False):
        super(ClassBlock, self).__init__()
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)] 
        #num_bottleneck=input_dim
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        f = self.add_block(x)
        f_norm = f.norm(p=2, dim=1, keepdim=True) + 1e-8
        f_normalize = f.div(f_norm)
        x = self.classifier(f)
        return x, f_normalize

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 8 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        
        #เปลี่ยนจาก (1,1) เป็น (self.part,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.model = model_ft
               
        #เปลี่ยนจาก (self.part,1) เป็น (1,1)
        self.dropout = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        #เพิ่มเข้ามา
        self.classifier = ClassBlock(2048, class_num, dropout=False, relu=False, num_bottleneck=2048, linear=True)
        
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, dropout=True, relu=False, num_bottleneck=256, linear=True))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) #[32,2048,24,12]
        
        #PCB part
        x_pcb = self.model.avgpool(x) #[32,2048,6,1]
        x_pcb = self.dropout(x_pcb) #[32,2048,6,1]
        
        #Triplet part
        x_triplet = self.avgpool(x) #[32,2048,1,1]
        x_triplet = self.dropout(x_triplet) #[32,2048,1,1]
        x_triplet = torch.squeeze(x_triplet) #[32,2048]
        x_triplet_iden, f_triplet = self.classifier(x_triplet) # f [32,2048]
                                                                             # x [32,781]
        
    
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = x_pcb[:,:,i].view(x_pcb.size(0), x_pcb.size(1)) #[32, 2048]
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i], _ = c(part[i]) # predict[i] have size [32,781] for each part feature input (6 parts in total)

        #print(predict)
        #print('x classification vector (32,781) of the first part-feature',predict[0][0].shape) 
        #print('f feature (32,2048) of the first part-feature',predict[0][1].shape)

        # sum prediction
        #y = predict[0]
        #for i in range(self.part-1):
        #    y += predict[i+1]
        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y, x_triplet_iden, f_triplet

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        
        #เพิ่มเข้ามา
        self.classifier = model.classifier

        self.avgpool1 = nn.AdaptiveAvgPool2d((self.part,1))
        
        #เพิ่มเข้ามา
        self.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
        
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) #[32,2048,24,12]
        
        #PCB part
        x_pcb = self.avgpool1(x) #[32,2048,6,1]
        y = x_pcb.view(x_pcb.size(0),x_pcb.size(1),x_pcb.size(2)) #[32,2048,6]

        #Triplet part
        x_tl = self.avgpool2(x) #[32,2048,1,1]
        x_tl = torch.squeeze(x_tl) #[32,2048]
        x1,f1 = self.classifier(x_tl) #x1=[32,781] , f1=[32,2048]
        
        return y, x1, f1
              
        # x_tl = x    #[32,2048,6,1]
        # y = x.view(x.size(0),x.size(1),x.size(2)) #[32,2048,6]
        
        # x_tl = self.avgpool2(x_tl) #[32,2048,1,1]
        # x_tl = torch.squeeze(x_tl) #[32,2048]
        # x1,f1 = self.classifier(x_tl) #x1=[32,781] , f1=[32,2048]
        # return y, x1, f1

# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num ):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, dropout=False, relu=False)
        # remove the final downsample
        # self.model.layer4[0].downsample[0].stride = (1,1)
        # self.model.layer4[0].conv2.stride = (1,1)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x,f = self.classifier(x)
        return x,f



# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.fc = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 1024 
        self.classifier = ClassBlock(1024, class_num)

    def forward(self, x):
        x = self.model.features(x)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num ):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = torch.squeeze(x)
        x = self.classifier(x)
        return x

# model = PCB(781)
# print(model)
# print(summary(model,(3,384,192), verbose = 0))


# debug model structure
#net = ft_net(751)
#net = ft_net(751)
#print(net)
#input = Variable(torch.FloatTensor(8, 3, 224, 224))
#output,f = net(input)
#print('net output size:')
#print(f.shape)

# model = PCB(781)
# print(model)
# # print(summary(model,(3,384,192), verbose = 0))
# inputs = Variable(torch.FloatTensor(32, 3, 384, 192))
# inputs = inputs
# outputs = model(inputs)
# print(outputs)
# print('outputs',outputs)
# print(outputs[0][0].shape)

# part = {}
# num_part = 6
# for i in range(num_part):
#     part[i] = outputs[i]

# print('part',part)

# sm = nn.Softmax(dim=1)
# score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
# print('score',score)
