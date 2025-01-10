import json
import matplotlib.pyplot as plt

def extract_loss_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get("g_losses", []), data.get("d_losses", [])

def plot_losses(g, d, legends=None, title="Loss Plot", xlabel="Epochs", ylabel="Loss", filename='loss'):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    for i in range(3):
        plt.plot(g[i], label=f'{legends[i]} Generator', color=colors[i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{filename}_gen.png')
    plt.close()

    for i in range(3):
        plt.plot(d[i], label=f'{legends[i]} Discriminator', color=colors[3+i])

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./{filename}_disc.png')


if __name__ == "__main__":
    json_file_paths = ["../results/celeba_gan_default_s10000_e50_bs32_glr0.0002_dlr5e-05.json",
                       "../results/celeba_gan_default_s50000_e50_bs64_glr0.0001_dlr0.0001.json",
                       "../results/celeba_gan_default_s75000_e100_bs64_glr0.0001_dlr5e-05.json"]
    g_losses, d_losses = [], []
    for file in json_file_paths:
        g_loss, d_loss = extract_loss_from_json(file)
        g_losses.append(g_loss[:50])
        d_losses.append(d_loss[:50])

    plot_losses(
        g_losses,
        d_losses,
        ['Exp. 1', 'Exp. 2', 'Exp. 3'],
        title="GAN Default Backbone Loss Plots",
        xlabel="Epochs",
        ylabel="Loss",
        filename='gan_loss'
    )

