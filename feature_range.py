from glob import glob
import profile

import numpy as np
import torch

torch.manual_seed(0)

import coder
from train import AverageMeter

def test(args, net, crop=None):
    # criterion = RateDistortionLoss()
    im, _, _ = coder.read_image(args.source)
    im = im.to(args.device)

    with torch.no_grad():
        y_main = net.g_a(im)
        index_max = torch.amax(y_main, dim=(2,3), keepdim=True)
        index_min = torch.amin(y_main, dim=(2,3), keepdim=True)
        index_abs_max = torch.amax(torch.abs(y_main), dim=(2,3), keepdim=True)
    return index_max, index_min, index_abs_max

if __name__ == "__main__":
    parser = coder.config()
    args = parser.parse_args()
    print("[Activation Range Evaluator]:", args.source)
    net = coder.load_model(args, training=False).to(args.device)

    bpp, psnr, msim, msim_dB = [AverageMeter() for i in range(4)]
    max_kodak, min_kodak = [AverageMeter() for i in range(2)]
    images = glob(args.source)
    index_max_, index_min_, index_abs_max_ = [], [], []
    for i, image in enumerate(images):
        print(i)
        if i > 10000:
            break
        args.source = image
        index_max, index_min, index_abs_max = test(args, net, crop=None)
        if index_max_ == None and index_min_ == None:
            index_max_, index_min_, index_abs_max_ = torch.zeros_like(index_max), torch.zeros_like(index_min), torch.zeros_like(index_abs_max)
        # print(index_max[0,:5,0,0], index_min[0,:5,0,0])
        # index_max_ = torch.where(index_max_ > index_max, index_max_, index_max)
        # index_min_ = torch.where(index_min_ < index_min, index_min_, index_min)
        index_max_.append(index_max)
        index_min_.append(index_min)
        index_abs_max_.append(index_abs_max)
        
    # average over index_max_
    index_max_ = torch.cat(index_max_, dim=0)
    index_max_mean = torch.mean(index_max_, dim=0, keepdim=True)
    # average over index_min_
    index_min_ = torch.cat(index_min_, dim=0)
    index_min_mean = torch.mean(index_min_, dim=0, keepdim=True)
    # get std of index_max_
    index_max_std = torch.std(index_max_, dim=0, keepdim=True)
    # get std of index_min_
    index_min_std = torch.std(index_min_, dim=0, keepdim=True)

    # boundry value over all images
    # channel_max = torch.amax(index_max_, dim=0)[:,0,0] 
    # channel_min = torch.amin(index_min_, dim=0)[:,0,0] # [C]
    # print(channel_max.size())
    channel_max = index_max_.topk(k=100, dim=0)[0][-1,:,0,0]
    channel_min = index_min_.topk(k=100, dim=0, largest=False)[0][-1,:,0,0]
    # print(channel_max.size())
    print(channel_max, channel_min)
    profile = f'{args.model}-{args.metric}-{args.quality}'
    if args.adv:
        profile += '-adv'
    torch.save([channel_max, channel_min], f"./attack/data/{profile}_range.pt")

    # print index of index_max where index_max_ is smaller than thres
    # dead_channels = []
    # thres = 2.0
    # for i, [v_max, v_min] in enumerate(zip(index_max_[0,:,0,0], index_min_[0,:,0,0])):
    #     if v_max < thres and v_min > -thres:
    #         print(i)
    #         dead_channels.append(i)
    # print(f"{len(dead_channels)}/{index_max.shape[1]} channels are dead")

    # index_abs_max_ = torch.cat(index_abs_max_, dim=0)
    # # channel_rank
    # ranks = []
    # for i in range(index_max_.shape[0]):
    #     rank = torch.argsort(index_abs_max_[i:i+1,:,0, 0], dim=1, descending=True)
    #     rank_ = torch.zeros_like(rank)
    #     for j in range(rank.shape[1]):
    #         # print(j,rank[0,j])
    #         rank_[0, rank[0,j]] = j
    #     ranks.append(rank_)
    # ranks = torch.cat(ranks, dim=0)
    # print(ranks.shape)
    # ranks_max = torch.amax(ranks, dim=0)
    # ranks_min = torch.amin(ranks, dim=0)
    # for i in range(ranks_min.shape[0]):
    #     print(i, ranks_min[i], ranks_max[i])
    # # print(index_max_mean[0,:10,0,0].cpu().numpy(), index_min_mean[0,:10,0,0].cpu().numpy())    
    # # print(index_max_std[0,:10,0,0].cpu().numpy(), index_min_std[0,:10,0,0].cpu().numpy())
    # # update max and min in index_max
    # # torch.save([index_max_mean+2*index_max_std, index_min_mean-2*index_min_std], f'{args.model}-{args.metric}-{args.quality}.pt')
    # torch.save({"dead":dead_channels, "rank": [ranks_max, ranks_min]}, f'{profile}.pt')