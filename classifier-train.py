import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

import classifier

def load_data(train_data_dir, train_batch_size):
    
    train_transform = transforms.Compose([
        #transforms.Resize(256),
        #transforms.RandomResizedCrop(size=112),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        train_data_dir, 
        transform=train_transform
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=train_batch_size, 
        shuffle=True,
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )
    return train_loader

def train():
    batch_size = 32
    ckpt_dir = f"./ckpts/attack/anchor/classifier"
    mnist_classifier = Classifier().cuda()
    optimizer = torch.optim.Adam(mnist_classifier.parameters(),lr=1e-3)
    # cross entropy loss
    # TODO: learning rate decay
    loss_func = nn.CrossEntropyLoss()
    # train loop
    for epoch in range(200):
        # input dataset
        loss_epoch =0.
        train_loader = load_data('/workspace/ct/code/mnist_png/training', batch_size)
        for step, (batch_x, targets) in enumerate(train_loader):
            batch_x = batch_x.to('cuda')
            # print(targets)
            batch_x = torch.reshape(batch_x, (batch_size, -1))
            output = mnist_classifier(batch_x)
            loss = loss_func(output, targets.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            print(epoch,step, loss.item())
        torch.save(mnist_classifier.state_dict(), os.path.join(ckpt_dir,'classifier_%d_%0.8f.pkl' % (epoch, loss_epoch/(step+1))))

def test(args):
    mnist_classifier = classifier.Classifier()
    mnist_classifier.load_state_dict(torch.load(args.ckpt))
    mnist_classifier = mnist_classifier.cuda()
    img = Image.open(args.source)
    img = np.array(img)/255.0
    if len(img.shape) < 3:
        H, W = img.shape
        img = np.tile(img.reshape((H,W,1)), (1,1,3))
    else:
        H, W, _ = img.shape
    im = torch.FloatTensor(img).cuda()
    im = im.permute(2, 0, 1).contiguous()
    im = torch.reshape(im, (1, -1))
    out = mnist_classifier(im)
    print(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test',dest='test',   action='store_true')
    parser.add_argument("-s",    dest="source", type=str, default=None, help="source input image")
    parser.add_argument("-ckpt", dest="ckpt",   type=str, default=None, help="local checkpoint dir")
    # python classifier.py
    args = parser.parse_args()
    # print("It is", args.test, "Mode!")
    if args.test:
        test(args)
    else:
        train()