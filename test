import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model_vlm import VL-YUV
from dataloader_vlms import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
import lpips

device_lpips = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lpips_model = lpips.LPIPS(net='alex').to(device_lpips)
lpips_model.eval()

def calculate_lpips_single(img1, img2):

    img1_lpips = (img1 - 0.5) * 2.0
    img2_lpips = (img2 - 0.5) * 2.0
    img1_lpips = img1_lpips.to(device_lpips)
    img2_lpips = img2_lpips.to(device_lpips)
    with torch.no_grad():
        lpips_val = lpips_model(img1_lpips, img2_lpips)
    return lpips_val.item()

def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()

        if mean_restored > 1e-5:
            img1 = img1 * (mean_target / mean_restored)
        img1 = torch.clamp(img1, 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()

def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            low, high, text_descs = batch
            low, high = low.to(device), high.to(device)

            output = model(low, text_descriptions=text_descs)
            output = torch.clamp(output, 0, 1)

            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))

            psnr = calculate_psnr(output, high)
            ssim = calculate_ssim(output, high)
            lpips_val = calculate_lpips_single(output, high)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val

            print(f"图像 {idx} | PSNR: {psnr:.4f} dB | SSIM: {ssim:.4f} | LPIPS: {lpips_val:.4f}")

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)
    return avg_psnr, avg_ssim, avg_lpips

def main():

    test_low = r'E:\VL-YUV-main\LOLv1\Eval15\low'
    test_high = r'E:\VL-YUV-main\LOLv1\Eval15\high'
    test_text = r'E:\VL-YUV-main\LOLv1\Eval15\text'
    weights_path = r'E:\VL-YUV-main\LOLv1_model.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = os.path.basename(os.path.dirname(test_low))
    result_dir = os.path.join('training_results_llava_LSRW', dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(
        None, None, test_low, test_high,
        train_text=None,
        test_text=test_text,
        crop_size=None,
        batch_size=1
    )

    print(f'测试数据集包含 {len(test_loader)} 张图像')

    model = VL_YUV().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'模型加载完成: {weights_path}')

    avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, result_dir)

    print(f'\n===== 测试完成 =====')
    print(f'数据集: {dataset_name}')
    print(f'平均 PSNR: {avg_psnr:.4f} dB')
    print(f'平均 SSIM: {avg_ssim:.4f}')
    print(f'平均 LPIPS: {avg_lpips:.4f}')

    return avg_psnr, avg_ssim, avg_lpips

if __name__ == '__main__':
    main()
