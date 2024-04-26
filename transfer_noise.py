from glob import glob

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

torch.manual_seed(0)

import coder
from train import RateDistortionLoss
from attack_rd import attack_


def plot_transferability_matrix(transfer_matrix, filename="transferability_matrix.pdf"):
    num_models = transfer_matrix.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(transfer_matrix, cmap='viridis', aspect='auto')

    # Add exact transferability numbers as text labels
    for i in range(num_models):
        for j in range(num_models):
            text = ax.text(j, i, f"{transfer_matrix[i, j]:.2f}",
                           ha="center", va="center", color="w", fontsize="x-large")

    # ax.set_xlabel("Target Model")
    # ax.set_ylabel("Source Model")

    ax.set_xticks(range(num_models))
    ax.set_yticks(range(num_models))
    # ax.set_xticklabels(["Balle2016", "Balle2018", "Minnen", "Cheng2020"])
    # ax.set_yticklabels(["Balle2016", "Balle2018", "Minnen", "Cheng2020"])
    ax.set_xticklabels(range(1, num_models+1))
    ax.set_yticklabels(range(1, num_models+1))
    
    # ax.set_title("Transferability Matrix")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
@torch.no_grad()
def test_transferability_multiple_models(models, images, args):
    num_models = len(models)
    transfer_matrix = np.zeros((num_models, num_models))

    for i, net1 in enumerate(models):
        # print(f"Generating adversarial examples for model {i + 1}/{num_models}...")
        # Generate adversarial examples for the current model (net1)
        adversarial_examples = []
        for idx, image in enumerate(images):
            # print(f"  Processing image {idx + 1}/{len(images)}")
            im_s, _, _ = coder.read_image(image)
            im_s = im_s.to(args.device)

            with torch.set_grad_enabled(True):
                im_adv, _, _, _, _, _, _ = attack_(im_s, net1, args)

            adversarial_examples.append([im_s, im_adv])

        for j, net2 in enumerate(models):
            print(f"Testing transferability from model {i + 1} to model {j + 1}")
            transfer_results = []

            for im_s, im_adv in adversarial_examples:
                output_adv2 = torch.clamp(net2(im_adv)["x_hat"], min=0., max=1.)
                output_s2 = torch.clamp(net2(im_s)["x_hat"], min=0., max=1.)
                mse_in = torch.mean((im_s - im_adv) ** 2)
                mse_out = torch.mean((output_adv2 - output_s2) ** 2)

                transferability = 10 * torch.log10(mse_out / mse_in)
                transfer_results.append(transferability.item())

            transfer_rate = np.mean(transfer_results)
            transfer_matrix[i, j] = transfer_rate

    return transfer_matrix


@torch.no_grad()
def test(net, image, noise, crop=None):
    # criterion = RateDistortionLoss()
    im, H, W = coder.read_image(image)
    im = im.to(noise.device)
    output_s = net(im)["x_hat"]
    im_ = torch.clamp(im+noise, min=0., max=1.)
    mse_in = torch.mean((im_ - im)**2)
    output_adv = net(im_)["x_hat"]
    if args.debug:
        coder.write_image(im_, "transfer_in.png", H, W)
        coder.write_image(output_adv, "transfer_out_noclamp.png", H, W)
        coder.write_image(torch.clamp(output_adv, min=0., max=1.), "transfer_out.png", H, W)
    # return criterion(result, im, training=False)
    return mse_in, torch.clamp(output_adv, min=0., max=1.), torch.clamp(output_s, min=0., max=1.)

if __name__ == "__main__":
    parser = coder.config()
    parser.add_argument("-s2", "--source2", type=str, default=None, help="source image")
    parser.add_argument("--cross-model", action="store_true", help="Flag for enabling cross-model transferability test")
    parser.add_argument("-model2", type=int, default=-1, help="Path to the second model's checkpoint file")
    args = parser.parse_args()
    
    
    # myattacker = attacker(args)

    # generate noise on source image A
    # bpp_ori, bpp, vi = myattacker.attack(args.source2)
    if args.source2:
        print("[Noise Transfer]:", args.source2, 'to' ,args.source)
        model_config = f"{args.model}_{args.quality}_{args.metric}_"
        net = coder.load_model(args, training=False).to(args.device)
        images_ = sorted(glob(args.source2))
        vis = torch.zeros(24,24)
        print(vis.shape)
        for i, source in enumerate(images_):
            im_s, H, W = coder.read_image(source)
            im_s = im_s.to(args.device)
            im_adv, output_adv, output_s, _, _, _, vi_results = attack_(im_s, net, args)
            noise = im_adv - im_s
            # mse_in = torch.mean(noise**2)
            # print(vi)
            
            coder.write_image(im_adv, "attack/major_tcsvt/before_transfer_in.png")
            coder.write_image(output_adv, "./attack/major_tcsvt/before_transfer_out.png")
            coder.write_image(noise+0.5, "./attack/major_tcsvt/noise.png")

            # add noise to source image B
            images = sorted(glob(args.source))
            for j, image in enumerate(images):
                mse_in, output_adv, output_s = test(net, image, noise)
                mse_out = torch.mean((output_adv - output_s)**2)
                print(10*math.log10(mse_out/mse_in))
                vis[i,j] = 10*math.log10(mse_out/mse_in)

                coder.write_image(output_adv, "./attack/major_tcsvt/after_transfer_out.png")
                # if args.debug:
                #     print(metrics)
            # print(metrics["bpp_loss"], metrics["mse_loss"], metrics["psnr"], metrics["msim_loss"], metrics["msim_dB"])
        torch.save(vis, model_config+"transfer.pkl")
        vis = torch.load(model_config+"transfer.pkl")
        # plt.show(vis)
        fig, ax = plt.subplots()
        im = ax.imshow(vis, vmin=-4, vmax=25)
        for i in range(24):
            for j in range(24):
                text = ax.text(j, i, int(vis[i, j].item()),
                            ha="center", va="center", color="w", fontsize="xx-small")
        plt.savefig(model_config+"transfer.pdf")
        print(vis)
    
    if args.cross_model:
        # images = sorted(glob(args.source))
        # # Load the set of models
        # if args.model2 >= 0:
        #     raise NotImplementedError("Loading models using the model2 argument is not yet implemented.")
        # else:
        #     models = []
        #     # for i in range(1, 7):
        #     #     args.quality = i
        #     #     models.append(coder.load_model(args, training=False).to(args.device))
            
        #     for model_ in ["factorized", "hyper", "context", "cheng2020"]:
        #         args.model = model_
        #         models.append(coder.load_model(args, training=False).to(args.device))
                
        # transfer_matrix = test_transferability_multiple_models(models, images, args)
        # print("Transferability matrix:")
        # print(transfer_matrix)

        # # Visualize the transferability matrix
        # np.save("transfer_methods.npy", transfer_matrix)
        
        transfer_matrix = np.load("transfer_methods.npy")
        plot_transferability_matrix(transfer_matrix=transfer_matrix)
    