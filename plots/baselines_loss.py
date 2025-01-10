import json
import matplotlib.pyplot as plt

def extract_loss_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data.get("test", [])

def plot_multiple_lists(data_lists, legends, title="Plot", xlabel="X-Axis", ylabel="Y-Axis", filename='loss'):

    for data in data_lists:
        plt.plot(data)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legends)
    plt.grid(True)
    plt.savefig(f'./{filename}.png')

# Example usage:
if __name__ == "__main__":
    # Extract loss values from a JSON file
    json_file_path = "../results/rbf_s10000_e100_bs16_lr0.001.json"
    rbf_loss = extract_loss_from_json(json_file_path)
    json_file_path = "../results/conv_s10000_e100_bs32_lr0.01.json"
    conv_loss = extract_loss_from_json(json_file_path)

    # Plot the loss values
    plot_multiple_lists(
        [rbf_loss, conv_loss],
        ["RBF", 'CONV'],
        title="Loss Plot",
        xlabel="Epochs",
        ylabel="Loss",
        filename='baselines_loss'
    )

