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

def plot_distrib_(ax, y_hat, y_hat_t, y_pred, y_pred_t, label, idx):
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
    ax.subplot(4,5,idx)
    ax.bar(x, y_hat_hist, label="gt")
    ax.plot(x_pred_fine, y_pred, label=f"pred", color="black")
    ax.legend(fontsize=8)
    # ax.tight_layout()
    ax.ylim(ymax=0.1)
    ax.title(f"channel-{label} (nature image)")
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

def plot_distrib(ax, y_hat, y_pred, label=None, title=None, x_title=None):
    x = np.linspace(v_min+0.5, v_max-0.5, bins)
    x_pred_fine = np.linspace(v_min+0.5, v_max-0.5, 50*bins)

    y_hat_hist = y_hat.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    ax.bar(x, y_hat_hist, label="gt")
    ax.plot(x_pred_fine, y_pred, label=f"pred", color="black")
    ax.legend(fontsize=4)
    ax.tick_params(axis='both', labelsize=4)
    # ax.tight_layout()
    ax.set_ylim(ymax=0.1)
    

    # ax.xaxis.set_label_position('top')
    if x_title:
        ax.set_xlabel(x_title, fontsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    # ax.set_title(f"channel-{label}", loc="left", x=0.02, y=1, pad=-20)
    ax.text(0.03, 0.95, f"channel-{label}", fontsize=4, ha='left', va='top', transform=ax.transAxes)
        
    ax.set_xticks(np.arange(min(x)+1, max(x), 2.0))
    # ax.autoscale(tight=True)

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
    upper = m1.cdf(ax_x+0.5)

    pred = torch.clamp(upper-lower, min=1./65536.)
    if dim:
        return torch.mean(pred, dim=dim)
    else:
        return pred

if __name__ == "__main__":
    parser = coder.config()
    parser.add_argument("-t2", dest="at_image", type=str)
    args = parser.parse_args()
    print(plt.rcParams['font.family'])
    from matplotlib.font_manager import findfont, FontProperties
    font = findfont(FontProperties(family=['sans-serif']))
    print(font)
    fontsize = 8

    with torch.no_grad():
        ckpt = args.checkpoint
        args.ckpt = None
        net = coder.load_model(args, training=False).to(args.device)
        scales_base_ori, y_base_ori = get_data(args.source, net)
        scales_base_adv, y_base_adv = get_data(args.target, net)
        if args.model in ["context", "cheng2020"]:
            im = coder.read_image(args.source)[0].to(args.device)
            means_base_ori = probe(im, net, "means_hat", MODEL=args.model)
            im = coder.read_image(args.target)[0].to(args.device)
            means_base_adv = probe(im, net, "means_hat", MODEL=args.model)

        args.checkpoint = ckpt
        net = coder.load_model(args, training=False).to(args.device)
        scales_at_ori, y_at_ori = get_data(args.source, net)
        scales_at_adv, y_at_adv = get_data(args.at_image, net)
        if args.model in ["context", "cheng2020"]:
            im = coder.read_image(args.source)[0].to(args.device)
            means_at_ori = probe(im, net, "means_hat", MODEL=args.model)
            im = coder.read_image(args.target)[0].to(args.device)
            means_at_adv = probe(im, net, "means_hat", MODEL=args.model)

        pixels = y_base_ori.shape[1] * y_base_ori.shape[2]
    if args.model == "hyper":
        means_base_adv, means_base_ori, means_at_adv, means_at_ori = [torch.zeros_like(y_base_ori) for i in range(4)]

    v_min = -5.5
    v_max = 5.5
    bins = int(v_max-v_min)

    x_pred = np.linspace(v_min+0.5, v_max-0.5, bins)
    x_pred_fine = np.linspace(v_min+0.5, v_max-0.5, 50*bins)

    y_pred_base_ori = predicted_distribution(x_pred, means_base_ori, scales_base_ori)
    y_pred_base_ori_fine = predicted_distribution(x_pred_fine, means_base_ori, scales_base_ori)
    y_pred_base_adv = predicted_distribution(x_pred, means_base_adv, scales_base_adv)
    y_pred_base_adv_fine = predicted_distribution(x_pred_fine, means_base_adv, scales_base_adv)

    y_pred_at_ori = predicted_distribution(x_pred, means_at_ori, scales_at_ori)
    y_pred_at_ori_fine = predicted_distribution(x_pred_fine, means_at_ori, scales_at_ori)
    y_pred_at_adv = predicted_distribution(x_pred, means_at_adv, scales_at_adv)
    y_pred_at_adv_fine = predicted_distribution(x_pred_fine, means_at_adv, scales_at_adv)

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
    #             "y_pred_t": y_pred_t}, f"./logs/distrib/distrib_channel_{label_subfigs = fig.subfigures(1, 2, wspace=0.07)}.pth")
    
    # plot 2*num_plot images
    num_plots = 3
    # fig = plt.figure(figsize=(20,8))
    fig, axs = plt.subplots(2,num_plots,figsize=(3.5, 2.2), constrained_layout=True, sharex=True, sharey=True) # 3.5 inch for IEEE Trans
    axs[0,0].set_ylabel(r"$\it{p}$ (nature image)", fontsize=fontsize)
    axs[1,0].set_ylabel(r"$\it{p}$ (adv example)", fontsize=fontsize)
    # subfigs = fig.subfigures(2, 1)
    if args.iter_x > 0:
        idx = args.iter_x
        print(idx, err[idx])
        plot_distrib(y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", "")
    else:
        # subfigs[0].supylabel("Baseline")
        # subfigs[1].supylabel("AT")
        for i in range(num_plots):
            idx = channel_index[i]
            print(idx, err[idx])
            if not args.adv:
                # plot_distrib(ax, y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)
                plot_distrib(axs[0,i], y_hat_hist[idx]/pixels, y_pred_base_ori_fine[idx], f"{idx}")
                plot_distrib(axs[1,i], y_hat_t_hist[idx]/pixels, y_pred_base_adv_fine[idx], f"{idx}", x_title="latent activation")
                plt.savefig("./logs/bitrate_baseline.pdf")
            else:
                y_at_ori_hist = torch.histc(y_at_ori[idx], bins=bins, min=v_min, max=v_max)
                y_at_adv_hist = torch.histc(y_at_adv[idx], bins=bins, min=v_min, max=v_max)
                plot_distrib(axs[0,i], y_at_ori_hist/pixels, y_pred_at_ori_fine[idx], f"{idx}")
                plot_distrib(axs[1,i], y_at_adv_hist/pixels, y_pred_at_adv_fine[idx], f"{idx}", x_title="latent activation")
                plt.savefig("./logs/bitrate_at.pdf")
    #     figs_baseline = subfigs[0].subfigures(2, 1)
    #     ax_baseline = figs_baseline[0].subplots(1, num_plots)
    #     for i, ax in enumerate(ax_baseline):
    #         idx = channel_index[i]
    #         print(idx, err[idx])
    #         # plot_distrib(ax, y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)
    #         plot_distrib(ax, y_hat_hist[idx]/pixels, y_pred_base_ori_fine[idx], f"{idx}")
    #     # subfigs[0].suptitle('Baseline', fontsize='x-large')
    #     # subfigs[0].tight_layout()

    #     ax_adv = figs_baseline[1].subplots(1, num_plots)
    #     for i, ax in enumerate(ax_adv):
    #         idx = channel_index[i]
    #         print(idx, err[idx])
    #         # plot_distrib(ax, y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)
    #         plot_distrib(ax, y_hat_t_hist[idx]/pixels, y_pred_base_adv_fine[idx], f"{idx}")
        
    #     figs_at = subfigs[1].subfigures(2, 1)
        
    #     ax_at_baseline = figs_at[0].subplots(1, num_plots)
    #     for i, ax in enumerate(ax_at_baseline):
    #         idx = channel_index[i]
    #         print(idx, err[idx])
    #         y_at_ori_hist = torch.histc(y_at_ori[idx], bins=bins, min=v_min, max=v_max)
    #         # plot_distrib(ax, y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)
    #         plot_distrib(ax, y_at_ori_hist/pixels, y_pred_at_ori_fine[idx], f"{idx}")

    #     ax_adv = figs_at[1].subplots(1, num_plots)
    #     for i, ax in enumerate(ax_adv):
    #         idx = channel_index[i]
    #         print(idx, err[idx])
    #         y_at_adv_hist = torch.histc(y_at_adv[idx], bins=bins, min=v_min, max=v_max)
    #         # plot_distrib(ax, y_hat_hist[idx]/pixels, y_hat_t_hist[idx]/pixels, y_pred_base_ori_fine[idx], y_pred_base_adv_fine[idx], f"{idx}", i+1)
    #         plot_distrib(ax, y_at_adv_hist/pixels, y_pred_at_adv_fine[idx], f"{idx}")
    #         plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    #     #     plot_distrib(y_at_ori_hist/pixels, y_at_adv_hist/pixels, y_pred_at_ori_fine[idx], y_pred_at_adv_fine[idx], f"{idx}", i+11)
    # # plt.show()
    # # subfigs[0].supylabel("Baseline")
    # # subfigs[1].supylabel("AT")
    # figs_baseline[0].supylabel("nature image", x=0.056)
    # figs_baseline[1].supylabel("adversarial example", x=0.056)
    # figs_at[0].supylabel("nature image", x=0.056)
    # figs_at[1].supylabel("adversarial example", x=0.056)
    # plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.8)
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