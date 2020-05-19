# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import time
import visdom

#https://blog.csdn.net/space_walk/article/details/102658047
def args_initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--GPU', default="0", help='GPU id')
    parser.add_argument('--continue_training', type=int, choices=[0,1], default=0, help='continue training')
    parser.add_argument('--latent_dim', type=int, default=100, help='array''s length to generate a girl')
    parser.add_argument('--test', choices=[0,1], type=int, default=0) 
    parser.add_argument('--train', choices=[0,1], type=int, default=0)
    parser.add_argument('--testmodelpath',default='checkpoint/Generator',help='test model path')
    parser.add_argument('--datapath',default='part_images',help='data image path')
    args = parser.parse_args()
    print(args)
    return args
    
##定义卷积核
def default_conv(in_channels,out_channels,kernel_size,bias=True):
    return nn.Conv2d(in_channels,out_channels,
                     kernel_size,padding=kernel_size//2,   #保持尺寸
                     bias=bias)
##定义ReLU     
def default_relu():
    return nn.ReLU(inplace=True)
## reshape
def get_feature(x):
    return x.reshape(x.size()[0],128,16,16)

class Generator(nn.Module):
    def __init__(self,input_dim=100,conv=default_conv,relu=default_relu,reshape=get_feature):
        super(Generator,self).__init__()
        head = [nn.Linear(input_dim,128*16*16),
                relu()]
        self.reshape = reshape                               #16x16
        body = [nn.Upsample(scale_factor=2,mode='nearest'),  #32x32
                conv(128,128,3),
                relu(),
                nn.Upsample(scale_factor=2,mode='nearest'),  #64x64
                conv(128,64,3),
                relu(),
                conv(64,3,3),
                nn.Tanh()]
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        
    def forward(self,x):#x:(batchsize,input_dim)
        x = self.head(x)
        x = self.reshape(x)
        x = self.body(x)
        return x        #(batchsize,3,64,64)
    def name(self):
        return 'Generator'

class Discriminator(nn.Module):
    def __init__(self,conv=default_conv,relu=default_relu):
        super(Discriminator,self).__init__()
        main = [conv(3,32,3),
                relu(),
                conv(32,64,3),
                relu(),
                conv(64,128,3),
                relu(),
                conv(128,256,3),
                relu()]
        self.main = nn.Sequential(*main)
        self.fc = nn.Linear(256*64*64,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):#x:(batchsize,3,64,64)
        x = self.main(x)#(b,256,64,64)
        x = x.view(x.size()[0],-1)#(b,256*64*64)
        x = self.fc(x) #(b,1)
        x = self.sigmoid(x)
        return x
    def name(self):
        return 'Discriminator'
    
class GAN(nn.Module):
    def __init__(self,args):
        super(GAN,self).__init__()
        self.img_size = 64
        self.channels = 3    
        self.latent_dim = args.latent_dim
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size
        self.cuda = args.cuda
        self.interval = 20 #每相邻20个epoch验证一次
        self.continue_training = args.continue_training #是否是继续训练        
        ## 生成器初始化
        self.generator = Generator(self.latent_dim)
        ## 判别器初始化
        self.discriminator = Discriminator()
        self.testmodelpath = args.testmodelpath
        self.datapath = args.datapath
        if self.cuda:
            self.generator.cuda()
            self.discriminator.cuda()
        self.continue_training_isrequired() 
        
    def trainer(self):
        ## 读入图片数据,分batch
        print('===> Data preparing...')
        import torchvision.transforms as transforms
        from torch.utils.data import DataLoader
        from torchvision.datasets import ImageFolder
        transform = transforms.ToTensor()  ##dataloader输出是tensor，不加这个会报错
        dataset = ImageFolder(self.datapath,transform=transform)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)       
        ##drop_last: dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
        num_batch = len(dataloader) #batch的数量为len(dataloader)=总图片数/batchsize
        print('num_batch:',num_batch)
        #dataloader: (batchsize,3,64,64) 分布0-1
        ## 判别值
        target_real = Variable(torch.ones(self.batch_size,1))
        target_false = Variable(torch.zeros(self.batch_size,1))
        one_const = Variable(torch.ones(self.batch_size,1))
        if self.cuda:
            target_real = target_real.cuda()
            target_false = target_false.cuda()
            one_const = one_const.cuda()
        ## 优化器
        optim_generator = optim.Adam(self.generator.parameters(),lr=0.0002,betas=(0.5,0.999))
        optim_discriminator = optim.Adam(self.discriminator.parameters(),lr=0.0002,betas=(0.5,0.999))
        ## 误差函数
#        content_criterion = nn.MSELoss()
        adversarial_criterion = nn.BCELoss()
        ## 训 练 开 始
        for epoch in range(self.start_epoch,self.num_epoch): ##epoch
            ##用于观察一个epoch不同batch的平均loss
            mean_dis_loss = 0.0
            mean_gen_con_loss = 0.0
            mean_gen_adv_loss = 0.0
            mean_gen_total_loss = 0.0
            for i,data in enumerate(dataloader):  ##循环次数：batch的数量为len(dataloader)=总图片数//batchsize
                if epoch<3 and i%10==0:
                    print('epoch%d: %d/%d'%(epoch,i,len(dataloader)))
                ##1.1生成noise
                gen_input = np.random.normal(0,1,(self.batch_size,self.latent_dim)).astype(np.float32)
                gen_input = torch.from_numpy(gen_input)
                gen_input = torch.autograd.Variable(gen_input,requires_grad=True)
                if self.cuda:
                    gen_input = gen_input.cuda()                    
                fake = self.generator(gen_input) ##生成器生成的(batchsize,3,64,64)
                real, _ = data  #data:list[tensor,tensor]取第零个 real:(batchsize,3,64,64)
                if self.cuda:
                    real = real.cuda()
                    fake = fake.cuda()
                ## 1.固定G,训练判别器D                
                self.discriminator.zero_grad()
                dis_loss1 = adversarial_criterion(self.discriminator(real),target_real)
                dis_loss2 = adversarial_criterion(self.discriminator(fake.detach()),target_false)##注意经过G的网络再进入D网络之前要detach()之后再进入
                dis_loss = 0.5*(dis_loss1+dis_loss2)
#                print('epoch:%d--%d,判别器loss:%.6f'%(epoch,i,dis_loss))
                dis_loss.backward()
                optim_discriminator.step()
                
                mean_dis_loss+=dis_loss
                ## 2.固定D,训练生成器G
                self.generator.zero_grad()
                ##生成noise
                gen_input = np.random.normal(0,1,(self.batch_size,self.latent_dim)).astype(np.float32)
                gen_input = torch.from_numpy(gen_input)
                gen_input = torch.autograd.Variable(gen_input,requires_grad=True)
                if self.cuda:
                    gen_input = gen_input.cuda()    
                fake = self.generator(gen_input) ##生成器生成的(batchsize,3,64,64)  
                gen_con_loss = 0
                gen_adv_loss = adversarial_criterion(self.discriminator(fake),one_const)##固定D更新G
                gen_total_loss = gen_con_loss + gen_adv_loss
#                print('epoch:%d--%d,生成器loss:%.6f'%(epoch,i,gen_total_loss))
                gen_total_loss.backward()
                optim_generator.step()
                mean_gen_con_loss+=gen_con_loss
                mean_gen_adv_loss+=gen_adv_loss
                mean_gen_total_loss+=gen_total_loss
                
            ## 一个epoch输出一次
            print('epoch:%d/%d'%(epoch, self.num_epoch))
            print('Discriminator_Loss: %.4f'%(mean_dis_loss/num_batch))
            print('Generator_total_Loss:%.4f'%(mean_gen_total_loss/num_batch))
            
            ## 保存模型
            state_dis = {'dis_model': self.discriminator.state_dict(), 'epoch': epoch}
            state_gen = {'gen_model': self.generator.state_dict(), 'epoch': epoch}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint') 
            torch.save(state_dis, 'checkpoint/'+self.discriminator.name()+'__'+str(epoch+1)) #each epoch
            torch.save(state_gen, 'checkpoint/'+self.generator.name()+'__'+str(epoch+1))     #each epoch
            torch.save(state_dis, 'checkpoint/'+self.discriminator.name())    #final  
            torch.save(state_gen, 'checkpoint/'+self.generator.name())        #final  
            ## 验证模型
            if epoch<45 or epoch%self.interval==0:
                 self.validater(epoch)
            print('--'.center(12,'-'))
            
    def validater(self,epoch):
        vis = visdom.Visdom(env='generate_girl_epoch%d'%(epoch))
        r,c = 3,3
        gen_input_val = np.random.normal(0,1,(r*c,self.latent_dim)).astype(np.float32)
        gen_input_val = torch.from_numpy(gen_input_val)
        gen_input_val = torch.autograd.Variable(gen_input_val)
        if self.cuda:
            gen_input_val = gen_input_val.cuda()   
        output_val = self.generator(gen_input_val)     #(r*c,3,64,64)
        output_val = output_val.cpu()
        output_val = output_val.data.numpy()      #(r*c,3,64,64)        
        img = np.transpose(output_val,(0,2,3,1))  #(r*c,64,64,3) 
        fig, axs = plt.subplots(r,c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                vis.image(output_val[cnt],opts={'title':'epoch%d_cnt%d'%(epoch,cnt)}) 
                axs[i, j].imshow(img[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1   
        if not os.path.isdir('images'):
            os.mkdir('images') 
        fig.savefig('images/val_%d.png'%(epoch+1)) ##保存验证结果
        plt.close()
    
    def tester(self,gen_input_test): #输入(N,latent_dim)
        assert gen_input_test.shape[1]==self.latent_dim, \
        'dimension 1''s size expect %d,but input %d'%(self.latent_dim,gen_input_test.shape[1])
        gen_input_test = gen_input_test.astype(np.float32)
        gen_input_test = torch.from_numpy(gen_input_test)
        gen_input_test = torch.autograd.Variable(gen_input_test)
        if self.cuda:
            gen_input_test = gen_input_test.cuda()   
        ## 下载验证结果
        if os.path.isdir('checkpoint'):
            try:
                checkpoint_gen = torch.load(self.testmodelpath)
                self.generator.load_state_dict(checkpoint_gen['gen_model'])
            except FileNotFoundError:
                print('Can\'t found dict')
        output_test = self.generator(gen_input_test)          
        output_test = output_test.cpu()
        output_test = output_test.data.numpy()      #(N,3,64,64)
        img = np.transpose(output_test,(0,2,3,1))  #(N,64,64,3) 
        if not os.path.isdir('images'):
            os.mkdir('images')         
        N = img.shape[0] #图像个数
        for i in range(N):
            plt.imshow(img[i, :, :, :])
            plt.axis('off')
            plt.savefig('images/test_%d.png'%(i+1)) ##保存结果
            plt.close()
                
    def continue_training_isrequired(self): ##是否在原基础上继续训练，定义start_epoch
        if self.continue_training == 1:
            print('===> Try resume from checkpoint')
            if os.path.isdir('checkpoint'):
                try:
                    checkpoint_dis = torch.load('checkpoint/'+self.discriminator.name())
                    self.discriminator.load_state_dict(checkpoint_dis['dis_model'])
                    checkpoint_gen = torch.load('checkpoint/'+self.generator.name())
                    self.generator.load_state_dict(checkpoint_gen['gen_model'])
                    self.start_epoch = checkpoint_dis['epoch']+1
                    print('===> Load last checkpoint data')
                except FileNotFoundError:
                    print('Can\'t found dict')
        elif self.continue_training == 0:
            self.start_epoch = 0    
            print('===> Start from scratch')
        else: 
            print('ERROR: wrong continue_training signal: neither 1 nor 0.')     
            
def main():
    start = time.clock()
    args = args_initialize()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU 
    if torch.cuda.is_available() and not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')            

    net = GAN(args)
    if args.train == 1:
        net.trainer()
    if args.test == 1:
        x = np.random.randint(0,1,(5,args.latent_dim))
        net.tester(x)
    
    elapsed = (time.clock() - start)
    print("Time used:",elapsed)    
    
if __name__=="__main__":
    main()
