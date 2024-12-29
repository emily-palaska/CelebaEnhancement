from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import torch, json

def save_metrics_to_json(metrics, file_path):
    """
    Save metrics dictionary to a JSON file.
    Converts NumPy types to Python native types to ensure JSON compatibility.
    """
    def convert_types(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Use a dictionary comprehension to convert values in the dictionary
    serializable_metrics = {key: convert_types(value) for key, value in metrics.items()}

    with open(file_path, 'w') as json_file:
        json.dump(serializable_metrics, json_file, indent=4)

def evaluate_image_quality(model, test_loader, device):
    """Evaluate image quality enhancement model.

    Args:
        model: The PyTorch model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run the model on.
        json_path: Path to save the metrics JSON file.

    Returns:
        A dictionary containing evaluation metrics.
    """
    model.eval()
    psnr_scores = []
    ssim_scores = []
    mse_scores = []

    with torch.no_grad():
        for degraded, clear in test_loader:
            degraded, clear = degraded.to(device), clear.to(device)
            outputs = model(degraded).cpu().numpy()
            clear = clear.cpu().numpy()

            for i in range(outputs.shape[0]):
                enhanced = outputs[i].transpose(1, 2, 0)  # Convert to HWC for metrics
                original = clear[i].transpose(1, 2, 0)    # Convert to HWC for metrics

                # Compute metrics for each image
                mse = np.mean((enhanced - original) ** 2)
                psnr_score = psnr(original, enhanced, data_range=original.max() - original.min())
                data_range = np.max(enhanced) - np.min(enhanced) + 1e-10 # Small number addition to avoid division by zero
                ssim_score = ssim(enhanced, original, channel_axis=2, data_range=data_range)

                mse_scores.append(mse)
                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)

    # Aggregate metrics
    metrics = {
        'avg_mse': np.mean(mse_scores),
        'avg_psnr': np.mean(psnr_scores),
        'avg_ssim': np.mean(ssim_scores),
    }

    return metrics
