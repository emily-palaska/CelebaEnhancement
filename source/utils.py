from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import numpy as np
import torch, json

def save_metrics_to_json(metrics, file_path):
    """Save metrics dictionary to a JSON file."""
    with open(file_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

def evaluate_image_quality(model, test_loader, device, json_path='./metrics.json'):
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
                ssim_score = ssim(original, enhanced, multichannel=True)

                mse_scores.append(mse)
                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)

    # Aggregate metrics
    metrics = {
        'avg_mse': np.mean(mse_scores),
        'avg_psnr': np.mean(psnr_scores),
        'avg_ssim': np.mean(ssim_scores),
    }

    # Save metrics to JSON file
    save_metrics_to_json(metrics, json_path)

    return metrics
