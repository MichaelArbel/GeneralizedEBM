from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os



def make_and_save_grid_images(images, name, samples_dir, N_h=8,N_w=8):
    N_tot = images.shape[0]
    tot= min(N_h*N_w, N_tot)
    samples = images[:tot].cpu().numpy()
    fig = plt.figure(figsize=(N_h, N_w))
    gs = gridspec.GridSpec(N_h, N_w)
    gs.update(wspace=0.05, hspace=0.05)
    images_list = []
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        sample_t = sample.transpose((1,2,0)) * 0.5 + 0.5
        images_list.append(sample_t)
        plt.imshow(sample_t)

    plt.savefig(
        os.path.join(samples_dir, f'{name}.png'),
        bbox_inches='tight')
    plt.close(fig)
    return images_list