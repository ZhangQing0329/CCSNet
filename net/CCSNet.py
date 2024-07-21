import torch
import torch.nn as nn
import torch.nn.functional as F
from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class feature_exctraction(nn.Module):
    def __init__(self, in_channel, depth, kernel):
        super(feature_exctraction, self).__init__()
        self.in_channel = in_channel
        self.depth = depth
        self.kernel = kernel
        self.conv1 = nn.Sequential(nn.Conv2d(self.depth, self.depth, self.kernel, 1, (self.kernel - 1) // 2),
                                   nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(self.in_channel, self.depth, 1, 1, 0), nn.BatchNorm2d(self.depth), nn.ReLU(inplace=True))

    def forward(self, x):

        conv1 = self.conv3(x)
        output = self.conv1(conv1)

        return output
              

    #def initialize(self):
     #   weight_init(self)

class SANet(nn.Module):

    def __init__(self, in_dim, coff=1):
        super(SANet, self).__init__()
        self.dim = in_dim
        self.coff = coff
        self.k = 9
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (1, self.k), 1, (0, self.k // 2)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_dim, self.dim // 4, (self.k, 1), 1, (self.k // 2, 0)), nn.BatchNorm2d(self.dim // 4), nn.ReLU(inplace=True))
        self.conv2_1 = nn.Conv2d(self.dim // 4, 1, (self.k, 1), 1, (self.k // 2, 0))
        self.conv2_2 = nn.Conv2d(self.dim // 4, 1, (1, self.k), 1, (0, self.k // 2))
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        conv1_1 = self.conv1_1(x)
        conv1_2 = self.conv1_2(x)
        conv2_1 = self.conv2_1(conv1_1)
        conv2_2 = self.conv2_2(conv1_2)
        conv3 = torch.add(conv2_1, conv2_2)
        conv4 = torch.sigmoid(conv3)

        conv5 = conv4.repeat(1, self.dim // self.coff, 1, 1)

        return conv5

    #def initialize(self):
     #   weight_init(self)

class SENet(nn.Module):

    def __init__(self, in_dim, ratio=2):
        super(SENet, self).__init__()
        self.dim = in_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, self.dim // ratio, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(self.dim // ratio, in_dim, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y=F.adaptive_avg_pool2d(x, (1,1)).view(b,c)
        y= self.sigmoid(self.fc(y)).view(b,c,1,1)
 
        output = y.expand_as(x)

        return output
        
class EdgeGenerate(nn.Module):

    def __init__(self):
        super(EdgeGenerate, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(64*2, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x2, x5):
        
        x5_up = F.upsample(x5, size=x2.size()[2:], mode='bilinear')
        output = self.conv1(torch.cat([x5_up, x2], dim=1))

        return output   
        
class EdgeRefine(nn.Module):

    def __init__(self):
        super(EdgeRefine, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(128, 2, 1, 1, 0), nn.BatchNorm2d(2))
        self.se = SENet(64)
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x, edge):
        
        edge_up = F.upsample(edge, size=x.size()[2:], mode='bilinear')
        edge_up = self.conv2(edge_up)
        x = self.conv3(x)
        wmap = self.conv1(torch.cat([edge_up, x], dim=1))
        wmap = F.softmax(wmap)
        wmap2 = torch.chunk(wmap, 2, dim=1)
        f1 = self.conv4(edge_up * wmap2[0] + x * wmap2[1])
        
        output = self.conv5(self.se(f1) * f1 + f1)

        return output, x * wmap2[1], edge_up * wmap2[0]              
        


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2,
                              bias=False)  # infer a one-channel attention map

    def forward(self, ftr):
        # ftr: [B, C, H, W]
        ftr_avg = torch.mean(ftr, dim=1, keepdim=True)  # [B, 1, H, W], average
        ftr_max, _ = torch.max(ftr, dim=1, keepdim=True)  # [B, 1, H, W], max
        ftr_cat = torch.cat([ftr_avg, ftr_max], dim=1)  # [B, 2, H, W]
        att_map = torch.sigmoid(self.conv(ftr_cat))  # [B, 1, H, W]
        return att_map

    #def initialize(self):
    #    weight_init(self)


class Fusion_3(nn.Module):

    def __init__(self):
        super(Fusion_3, self).__init__()
        
        self.ca1 = SENet(64)
        self.ca2 = SENet(64)
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.sa1 = SANet(64)
        self.sa2 = SANet(64)
        
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64*3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        

    def forward(self, x0, x1, x2): # x1 and x2 are the adjacent features
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')
        x2_up = F.upsample(x2, size=x0.size()[2:], mode='bilinear')
               
        x1_w = self.ca1(x1_up)
        x2_w = self.ca2(x2_up)

        se1 = x0 * x1_w
        se2 = x0 * x2_w
        
        x3_w = self.sa1(x1_up)
        x4_w = self.sa2(x2_up)
        
        fea1 = x0 * x3_w
        fea2 = x0 * x4_w
        
        fea3 = self.conv4(torch.cat([fea1, se1], dim=1))
        fea4 = self.conv5(torch.cat([fea2, se2], dim=1))
                      
        output = x0 + fea3 + fea4
        
        return output, fea4 
        
class Fusion_2(nn.Module):

    def __init__(self):
        super(Fusion_2, self).__init__()
        self.ca1 = SENet(64)        
        self.sa1 = SANet(64)               
        self.conv1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64*2, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        

    def forward(self, x0, x1): # x1 is the adjacent features
        x1_up = F.upsample(x1, size=x0.size()[2:], mode='bilinear')       

        x1_w = self.ca1(x1_up)       
        se1 = x0 * x1_w              
        x3_w = self.sa1(x1_up)       
        fea1 = x0 * x3_w
        
        fea3 = self.conv4(torch.cat([fea1, se1], dim=1))
              
        output = x0 + fea3
        
        return output
        
class ResidualLearning(nn.Module):

    def __init__(self, channel=64):
        super(ResidualLearning, self).__init__()
                       
        self.decoder5_1 = BasicConv2d(channel*2, 64, 3, padding=1)
        self.decoder5_2 = nn.Sequential(
            BasicConv2d(64, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.up5 = TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
       
        self.S5 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder4_1 = BasicConv2d(channel*2, 64, 3, padding=1)      
        self.decoder4_2 = nn.Sequential(
            BasicConv2d(64*2, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.up4 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.S4 = nn.Conv2d(64, 1, 3, stride=1, padding=1)

        self.decoder3_1 = BasicConv2d(channel*2, 64, 3, padding=1)      
        self.decoder3_2 = nn.Sequential(
            BasicConv2d(64*2, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.up3 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
        
        self.S3 = nn.Conv2d(64, 1, 3, stride=1, padding=1)  

        self.decoder2_1 = BasicConv2d(channel*2, 64, 3, padding=1)      
        self.decoder2_2 = nn.Sequential(
            BasicConv2d(64*2, 64, 3, padding=1),
            BasicConv2d(64, 64, 3, padding=1))
            #nn.Dropout(0.5),
        self.up2 =  TransBasicConv2d(64, 64, kernel_size=2, stride=2,
                              padding=0, dilation=1, bias=False)
              
        self.S2 = nn.Conv2d(64, 1, 3, stride=1, padding=1) 
        self.S6 = nn.Conv2d(64, 1, 3, stride=1, padding=1) 
        
        self.edge2_fusion =  EdgeRefine() 
        self.edge3_fusion =  EdgeRefine()
        self.edge4_fusion =  EdgeRefine()
        self.edge5_fusion =  EdgeRefine()                 

    def forward(self, x2, x3, x4, x5, edge, x6):  # x1 is the saliency map from the adjacent deep layer

        x6_up5 = F.upsample(x6, size=x5.size()[2:], mode='bilinear')
        x5_f = self.decoder5_1(torch.cat((x5, x6_up5), 1)) 
        x5_output, ref5o, ref5e = self.edge5_fusion(x5_f, edge)
        x5_cat = self.decoder5_2(x5_output)
        s5 = self.S5(x5_cat)
        x5_up = self.up5(x5_cat)
        
        x6_up4 = F.upsample(x6, size=x4.size()[2:], mode='bilinear')
        x4_f = self.decoder4_1(torch.cat((x4, x6_up4), 1)) 
        x4_output, ref4o, ref4e = self.edge4_fusion(x4_f, edge)
        x4_cat = self.decoder4_2(torch.cat((x4_output, x5_up),1))
        s4 = self.S4(x4_cat)       
        x4_up = self.up4(x4_cat)
        
        x6_up3 = F.upsample(x6, size=x3.size()[2:], mode='bilinear')        
        x3_f = self.decoder3_1(torch.cat((x3, x6_up3), 1))
        x3_output, ref3o, ref3e = self.edge3_fusion(x3_f, edge)
        x3_cat = self.decoder3_2(torch.cat((x3_output, x4_up),1))
        s3 = self.S3(x3_cat)       
        x3_up = self.up3(x3_cat)
        
        x6_up2 = F.upsample(x6, size=x2.size()[2:], mode='bilinear')   
        x2_f = self.decoder2_1(torch.cat((x2, x6_up2), 1)) 
        x2_output, ref2o, ref2e = self.edge2_fusion(x2_f, edge)
        x2_cat = self.decoder2_2(torch.cat((x2_output, x3_up),1))
        s2 = self.S2(x2_cat)
        
        s6 = self.S6(x6)       
    
                                
        return s2, s3, s4, s5, s6, x3_f, ref3e




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        self.fusion4 = Fusion_3()
        self.fusion3 = Fusion_3()
        self.fusion2 = Fusion_2()
        self.fusion5 = Fusion_2()

        
        self.fem_layer5 = feature_exctraction(512, 64, 7)
        self.fem_layer4 = feature_exctraction(512, 64, 5)
        self.fem_layer3 = feature_exctraction(256, 64, 5)
        self.fem_layer2 = feature_exctraction(128, 64, 3)
        self.fem_layer1 = feature_exctraction(64, 64, 3)

        self.con1 = nn.Sequential(nn.Conv2d(64, 64, 1,1, 0),nn.BatchNorm2d(64),nn.ReLU(inplace = True))
        self.con2 = nn.Sequential(nn.Conv2d(256, 128, 1,1, 0),nn.BatchNorm2d(128),nn.ReLU(inplace = True))
        self.con3 = nn.Sequential(nn.Conv2d(512, 256, 1,1, 0),nn.BatchNorm2d(256),nn.ReLU(inplace = True))
        self.con4 = nn.Sequential(nn.Conv2d(1024, 512,1,1, 0),nn.BatchNorm2d(512),nn.ReLU(inplace = True))
        self.con5 = nn.Sequential(nn.Conv2d(2048, 512,1,1,  0),nn.BatchNorm2d(512),nn.ReLU(inplace = True))
        
        self.edge_generate = EdgeGenerate()
        
        self.fem_layer5_e = feature_exctraction(512, 64, 7)
        self.fem_layer2_e = feature_exctraction(128, 64, 3)
        
        
        self.res1 = ResidualLearning()

        self.pooling = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(2048, 64, 1, 1, 0), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 7, 1, 3), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        
        self.addconv5 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.addconv4 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.addconv3 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.addconv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True), nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace =True))
        self.edgeconv = nn.Conv2d(64, 1, 1, 1, 0)

    # def initialize_weights(self):
    # model_state = torch.load('./models/resnet50-19c8e357.pth')
    # self.resnet.load_state_dict(model_state, strict=False)

    def forward(self, x):
        x2, x3, x4, x5 = self.resnet(x)
        
        x_size = x.size()

        conv2 = self.con2(x2)
        conv3 = self.con3(x3)
        conv4 = self.con4(x4)
        conv5 = self.con5(x5)
        
        conv6 = self.conv3(self.pooling(x5))
        

        fem_layer5 = self.fem_layer5(conv5)
        fem_layer4 = self.fem_layer4(conv4)
        fem_layer3 = self.fem_layer3(conv3)
        fem_layer2 = self.fem_layer2(conv2)

        edge_layer5 = self.fem_layer5_e(conv5)
        edge_layer2 = self.fem_layer2_e(conv2)
        
        edgefea = self.edge_generate(edge_layer2, edge_layer5)
        edgefea2 = self.edgeconv(edgefea)
        edgeatt = torch.sigmoid(edgefea2)

        # cross-attention
        f_sca4,fa = self.fusion4(fem_layer4, fem_layer3, fem_layer5)
        fea4 = self.addconv4(f_sca4)
        f_sca3,fb =  self.fusion3(fem_layer3, fem_layer2, fem_layer4)
        fea3 = self.addconv3(f_sca3)
        fea2 = self.addconv2(self.fusion2(fem_layer2, fem_layer3))
        fea5 = self.addconv5(self.fusion5(fem_layer5, fem_layer4))
        

        s2, s3, s4, s5, s6, ro, re = self.res1(fea2, fea3, fea4, fea5, edgefea, conv6)
        
        
        pre6 = F.upsample(edgeatt, size=x.size()[2:], mode='bilinear')
        pre5 = F.upsample(s6, size=x.size()[2:], mode='bilinear')
        pre4 = F.upsample(s5, size=x.size()[2:], mode='bilinear')
        pre3 = F.upsample(s4, size=x.size()[2:], mode='bilinear')
        pre2 = F.upsample(s3, size=x.size()[2:], mode='bilinear')
        pre1 = F.upsample(s2, size=x.size()[2:], mode='bilinear')


        return pre1, pre2, pre3, pre4, pre5, pre6
