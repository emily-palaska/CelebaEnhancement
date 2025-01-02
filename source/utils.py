from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import torch, json
from torch.utils.data import Dataset
from torchvision.transforms import Resize

def save_metrics_to_json(metrics, file_path):
    """
    Save metrics dictionary to a JSON file.
    """
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

def evaluate_image_quality(model, test_loader, device,  title='Evaluation of Image Enhancement'):
    """Evaluate image quality enhancement model.

    Args:
        model: The PyTorch model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on.
        title: Title of the plot

    Returns:
        A dictionary containing evaluation metrics.
    """
    model.eval()
    psnr_scores = []
    ssim_scores = []
    mse_scores = []
    enhanced_images = []
    original_images = []

    with torch.no_grad():
        for degraded, clear in test_loader:
            degraded, clear = degraded.to(device), clear.to(device)
            outputs = model(degraded).cpu().numpy()
            clear = clear.cpu().numpy()

            for i in range(outputs.shape[0]):
                enhanced = outputs[i].transpose(1, 2, 0)  # Convert to HWC for metrics
                original = clear[i].transpose(1, 2, 0)    # Convert to HWC for metrics

                if len(enhanced_images) < 10: # sample first 10 to plot them
                    enhanced_images.append(enhanced)
                    original_images.append(original)

                # Compute metrics for each image
                mse = np.mean((enhanced - original) ** 2)
                psnr_score = psnr(original, enhanced, data_range=original.max() - original.min() + 1e-10)
                data_range = np.max(enhanced) - np.min(enhanced) + 1e-10 # Small number addition to avoid division by zero
                ssim_score = ssim(enhanced, original, channel_axis=2, data_range=data_range)
                
                mse_scores.append(mse)
                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)
    # Plot examples of predictions
    enhanced_images = np.array(enhanced_images)
    enhanced_images = (enhanced_images - enhanced_images.min()) / (enhanced_images.max() - enhanced_images.min())
    print(np.min(enhanced_images), np.max(enhanced_images))
    plot_examples(original_images, enhanced_images, save_path='../plots/examples.png', title=title)

    # Aggregate metrics
    metrics = {
        'avg_mse': float(np.mean(mse_scores)),
        'avg_psnr': float(np.mean(psnr_scores)),
        'avg_ssim': float(np.mean(ssim_scores)),
    }
    return metrics



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
    
def plot_examples(x, y, num_examples=6, save_path='examples.png', title='Examples of CelebA Image Quality Enhacement Dataset'):
    """
    Plot random examples of give x and y
    """
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
