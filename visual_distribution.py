# from coder import test, config
from cProfile import label
import math
import torch
import numpy as np
from glob import glob
# from utils import visualizer
import coder
import scipy.stats as stats
from anchors.model import probe
from matplotlib import pyplot as plt

def plot_distrib(y_hat, y_hat_t, y_pred, y_pred_t, label, idx):
    # sigma = math.sqrt(scale_hat)
    # sigma = i.item()
    # v_max = math.ceil(torch.max(y_hat))+0.5
    # v_min = math.floor(torch.min(y_hat))-0.5
    # v_max = math.ceil(torch.max(y_hat))+0.5 if math.ceil(torch.max(y_hat))+0.5 > 5.5 else 5.5
    # v_min = math.floor(torch.min(y_hat))-0.5 if math.floor(torch.min(y_hat))-0.5 < -5.5 else -5.5
    # bins = int(v_max-v_min)
    # y_hat_hist = torch.histc(y_hat, bins=bins, min=v_min, max=v_max).cpu().numpy()/pixels
    
    # ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)
    
    y_hat_hist = y_hat.cpu().numpy()
    y_hat_t_hist = y_hat_t.cpu().numpy()
    # print(y_hat_hist)
    x = np.linspace(v_min+0.5, v_max-0.5, bins)
    x_pred_fine = np.linspace(v_min+0.5, v_max-0.5, 100*bins)
    # print(v_min,v_max,x)
    # y_l = stats.norm.cdf(x_pred-0.5, mu, sigma)
    # y_r = stats.norm.cdf(x_pred+0.5, mu, sigma)
    # y_pred = (y_r-y_l)
    y_pred = y_pred.cpu().numpy()
    y_pred_t = y_pred_t.cpu().numpy()
    plt.subplot(4,5,idx)
    plt.bar(x, y_hat_hist, label="gt")
    plt.plot(x_pred_fine, y_pred, label=f"pred", color="black")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.ylim(ymax=0.1)
    plt.title(f"channel-{label} (nature image)")
    # plt.savefig(f"./logs/distrib/image_ori_channel-{label}.pdf")
    # plt.close()
    plt.subplot(4,5,idx+5)
    plt.bar(x, y_hat_t_hist, label="gt")
    plt.plot(x_pred_fine, y_pred_t, label=f"pred", color="black")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.ylim(ymax=0.1)
    plt.title(f"channel-{label} (adv example)")
    # plt.savefig(f"./logs/distrib/image_adv_channel-{label}.pdf")
    # plt.close()

def get_data(file, net):
    im = coder.read_image(file)[0].to(args.device)
    return probe(im, net, "scales_hat")[0], probe(im, net, "y_hat")[0]
    
def predicted_distribution(ax_x, means, scales, dim=(1,2)):
    scales = torch.clamp(scales, min=0.11)
    bins = len(ax_x)
    ax_x = torch.tensor(ax_x).tile(scales.shape[0], scales.shape[1], scales.shape[2], 1).cuda()
    
    means = means.view(scales.shape[0], scales.shape[1], scales.shape[2], 1).repeat(1,1,1,bins)
    scales = scales.view(scales.shape[0], scales.shape[1], scales.shape[2],1).repeat(1,1,1,bins)
    
    m1 = torch.distributions.Normal(means, scales)
    lower = m1.cdf(ax_x-0.5)
    upper = m1.cdf(ax_x+0.5)\

    pred = torch.clamp(upper-lower, min=1./65536.)
    if dim:
        return torch.mean(pred, dim=dim)
    else:
        return pred

if __name__ == "__main__":
    parser = coder.config()
    parser.add_argument("-t2", dest="at_image", type=str)
    args = parser.parse_args()

    with torch.no_grad():
        ckpt = args.checkpoint
        args.ckpt = None
        net = coder.load_model(args, training=False).to(args.device)
        scales_base_ori, y_base_ori = get_data(args.source, net)
        scales_base_adv, y_base_adv = get_data(args.target, net)

        args.checkpoint = ckpt
        net = coder.load_model(args, training=False).to(args.device)
        scales_at_ori, y_at_ori = get_data(args.source, net)
        scales_at_adv, y_at_adv = get_data(args.at_image, net)

        pixels = y_base_ori.shape[1] * y_base_ori.shape[2]
    if args.model == "hyper":
        means = torch.zeros_like(y_base_ori)
    v_min = -5.5
    v_max = 5.5
    bins = int(v_max-v_min)

    x_pred = np.linspace(v_min+0.5, v_max-0.5, bins)
    x_pred_fine = np.linspace(v_min+0.5, v_max-0.5, 100*bins)

    y_pred_base_ori = predicted_distribution(x_pred, means, scales_base_ori)
    y_pred_base_ori_fine = predicted_distribution(x_pred_fine, means, scales_base_ori)
    y_pred_base_adv = predicted_distribution(x_pred, means, scales_base_adv)
    y_pred_base_adv_fine = predicted_distribution(x_pred_fine, means, scales_base_adv)

    y_pred_at_ori = predicted_distribution(x_pred, means, scales_at_ori)
    y_pred_at_ori_fine = predicted_distribution(x_pred_fine, means, scales_at_ori)
    y_pred_at_adv = predicted_distribution(x_pred, means, scales_at_adv)
    y_pred_at_adv_fine = predicted_distribution(x_pred_fine, means, scales_at_adv)

    err, y_hat_hist, y_hat_t_hist = [], [], []
    for i in range(y_base_ori.shape[0]):
        hist_ori = torch.histc(y_base_ori[i], bins=bins, min=v_min, max=v_max)
        hist_adv = torch.histc(y_base_adv[i], bins=bins, min=v_min, max=v_max)
        y_hat_hist.append(hist_ori)
        y_hat_t_hist.append(hist_adv)
        # print(y_pred[i])
        rate_ori = torch.sum(hist_ori * -torch.log(y_pred_base_ori[i]))/math.log(2)
        rate_adv = torch.sum(hist_adv * -torch.log(y_pred_base_adv[i]))/math.log(2)
        # print(rate_ori, rate_adv)
        
        # find the channel with maximum bitrate distance
        # pixel-wise comparison instead of channel wise

        err.append(rate_adv - rate_ori)
        # print(err[i])
        # y_hat_hist = y_hat_hist.cpu().numpy()
        # y_hat_t_hist = y_hat_t_hist.cpu().numpy()
    # sort y_hat_hist with err
    channel_index = np.arange(0, y_base_ori.shape[0])
    err = torch.stack(err)
    y_hat_hist = torch.stack(y_hat_hist)
    y_hat_t_hist = torch.stack(y_hat_t_hist)
    # y_hat_hist = y_hat_hist[err.argsort(descending=True)]
    # y_hat_t_hist = y_hat_t_hist[err.argsort(descending=True)]
    channel_index = channel_index[err.argsort(descending=True).cpu()]
    # err = err[err.argsort(descending=True)]
    if args.adv:
        label_ = "at"
    else:
        label_ = "baseline"
    # torch.save({"y_hat": y_hat_hist/pixels, 
    #             "y_hat_t_hist": y_hat_t_hist/pixels,
    #             "channel_index": channel_index, 
    #             "err": err,
    #             "y_pred": y_pred,
    #             "y_pred_t": y_pred_t}, f"./logs/distrib/distrib_channel_{label_}.pth")
    # plot 4x5 images
    plt.figure(figsize=(16,8))
    if args.iter_x > 0:
        idx = args.iter_x
        print(idx, err[idx])
        plot_distrib(y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}")
    else:    
        for i in range(5):
            idx = channel_index[i]
            print(idx, err[idx])
            plot_distrib(y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)

            y_at_ori_hist = torch.histc(y_at_ori[idx], bins=bins, min=v_min, max=v_max)
            y_at_adv_hist = torch.histc(y_at_adv[idx], bins=bins, min=v_min, max=v_max)
            plot_distrib(y_at_ori_hist/pixels, y_at_adv_hist/pixels, y_pred_at_ori_fine[idx], y_pred_at_adv_fine[idx], f"{idx}", i+11)

    plt.savefig("./logs/bitrate.pdf")
    # parser = coder.config()
    # args = parser.parse_args()
    # images = glob(args.source)
    # bins = 20
    # nums = 200
    # hist_ori = np.zeros((192,bins))
    # hist_adv = np.zeros((192,bins))
    # cnt = 0
    # im_num = len(images)
    # for image in images:
    #     cnt += 1
    #     print(cnt,"/",im_num)
        
    #     # adversarial
    #     args.source = image
    #     print("[adv]", args.source)
    #     _, _, _, y_main_, _ = test(args, checkpoint_dir=None)
    #     y_main_q = torch.round(y_main_).cpu().numpy()
        
    #     for i in range(192):
    #         hist, bin_edges = np.histogram(y_main_q[0,i,:,:], bins=bins, range=(-10,10), normed=None, weights=None, density=None)
    #         hist_adv[i] += hist
        
        
    #     # original
    #     adv_dir = "/workspace/ct/datasets/datasets/div2k/"
    #     args.source = adv_dir + image.split('/')[-1]
    #     print("[ori]", args.source)
    #     _, _, _, y_main_, _ = test(args, checkpoint_dir=None)
    #     y_main_q = torch.round(y_main_).cpu().numpy()
        
    #     for i in range(192):
    #         hist, bin_edges = np.histogram(y_main_q[0,i,:,:], bins=bins, range=(-10,10), normed=None, weights=None, density=None)
    #         hist_ori[i] += hist

    #     if cnt == nums:
    #         hist_ori = hist_ori/nums
    #         hist_adv = hist_adv/nums
    #         # for i in range(192):
    #         #     print(hist_all[i], hist_all[i]/100)
    #         break
            
    
    # # for i in range(192):
    # #     print(hist_all[i])
    # np.save("distrib_ori", hist_ori)
    # np.save("distrib_adv", hist_adv)

    # python visual_distribution.py -s /workspace/ct/datasets/kodak/kodim03.png -t ./attack/kodak/hyper_5_ms-ssim_kodim03_advin_baseline.png -t2 ./attack/kodak/hyper_5_ms-ssim_kodim03_advin_at.png -q 5