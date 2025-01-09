from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import torch, json
from torch.utils.data import Dataset
from torchvision.transforms import Resize

def save_metrics_to_json(metrics, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

def evaluate_image_quality(model, test_loader, device):
    model.eval()
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    x_samples = []
    y_samples = []
    y_hat_samples = []

    with torch.no_grad():
        for degraded, clear in test_loader:
            degraded, clear = degraded.to(device), clear.to(device)
            outputs, clear, degraded  = model(degraded).cpu().numpy(), clear.cpu().numpy(), degraded.cpu().numpy()

            for i in range(outputs.shape[0]):
                # Convert to HWC for metrics
                y_hat = outputs[i].transpose(1, 2, 0)
                y = clear[i].transpose(1, 2, 0)
                x = degraded[i].transpose(1, 2, 0)

                # Sample first 10 images to return
                if len(x_samples) < 10:
                    x_samples.append(x)
                    y_samples.append(y)
                    y_hat_samples.append(y_hat)

                # Compute metrics for each image
                mse = np.mean((y_hat - y) ** 2)
                psnr_score = psnr(y, y_hat, data_range=y.max() - y.min() + 1e-10)
                ssim_score = ssim(y_hat, y, channel_axis=2, data_range=y.max() - y.min() + 1e-10)
                
                mse_scores.append(mse)
                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)

    # Plot examples of predictions
    x_samples, y_samples, y_hat_samples = np.array(x_samples), np.array(y_samples), np.array(y_hat_samples)
    y_hat_samples = (y_hat_samples - y_hat_samples.min()) / (y_hat_samples.max() - y_hat_samples.min())

    # Aggregate metrics
    metrics = {
        'avg_mse': float(np.mean(mse_scores)),
        'avg_psnr': float(np.mean(psnr_scores)),
        'avg_ssim': float(np.mean(ssim_scores)),
    }
    return metrics, x_samples, y_samples, y_hat_samples


# Custom Dataset Class
class ImageEnhancementDataset(Dataset):
    def __init__(self, x_data, y_data, resize=False):
        self.x_data = torch.tensor(x_data, dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC to NCHW
        self.y_data = torch.tensor(y_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.resize = resize
        self.resize_tr = Resize((180, 220))  # Match conv model's output size

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.resize_tr(self.y_data[idx]) if self.resize else self.y_data[idx]
        return x, y

    def __len__(self):
        return len(self.x_data)
    
def plot_examples(x, y, num_examples=6, save_path='examples.png', title='Examples of CelebA Image Quality Enhancement Dataset'):
    # Scale for plotting
    x = x * 255 if np.max(x) <= 1.0 else x
    y = y * 255 if np.max(y) <= 1.0 else y

    fig, axes = plt.subplots(2, num_examples, figsize=(15, 5))
    fig.suptitle(title, fontsize=16)
    random_indices = np.random.choice(len(x), num_examples, replace=False)

    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(x[idx].astype(np.uint8))
        axes[0, i].axis('off')
        axes[0, i].set_title("x")

        axes[1, i].imshow(y[idx].astype(np.uint8))
        axes[1, i].axis('off')
        axes[1, i].set_title("y")

    plt.tight_layout()
    plt.savefig(save_path)

def plot_examples_with_predictions(x, y, y_hat, num_examples=6, save_path='enhanced_examples.png',
                                   title='Image Quality Enhancement Results'):
    x = x * 255 if np.max(x) <= 1.0 else x
    y = y * 255 if np.max(y) <= 1.0 else y
    y_hat = y_hat * 255 if np.max(y_hat) <= 1.0 else y_hat

    fig, axes = plt.subplots(3, num_examples, figsize=(15, 7))
    fig.suptitle(title, fontsize=16)
    random_indices = np.random.choice(len(x), num_examples, replace=False)

    for i, idx in enumerate(random_indices):
        axes[0, i].imshow(x[idx].astype(np.uint8))
        axes[0, i].axis('off')
        axes[0, i].set_title("x")

        axes[1, i].imshow(y[idx].astype(np.uint8))
        axes[1, i].axis('off')
        axes[1, i].set_title("y")

        axes[2, i].imshow(y_hat[idx].astype(np.uint8))
        axes[2, i].axis('off')
        axes[2, i].set_title("y_hat")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(save_path)
    plt.close(fig)