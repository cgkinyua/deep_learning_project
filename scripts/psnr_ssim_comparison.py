import pandas as pd
import matplotlib.pyplot as plt

# Load baseline results
baseline_results = pd.read_csv("C:/Users/USER/Desktop/Latest projects/deep_learning_project/results/simplified_baseline_evaluation_metrics.csv")
print("Baseline Results:")
print(baseline_results)

# Load enhanced results
enhanced_results = pd.read_csv("C:/Users/USER/Desktop/Latest projects/deep_learning_project/results/extended_evaluation_results_dynamic_contraction_augadvanced.csv")
print("\nEnhanced Results:")
print(enhanced_results)

# Calculate average PSNR and SSIM for baseline
baseline_psnr = baseline_results['PSNR (dB)'].mean()
baseline_ssim = baseline_results['SSIM'].mean()

# Calculate average PSNR and SSIM for enhanced model
enhanced_psnr = enhanced_results['PSNR'].mean()
enhanced_ssim = enhanced_results['SSIM'].mean()

print(f"Average Baseline PSNR: {baseline_psnr}, SSIM: {baseline_ssim}")
print(f"Average Enhanced PSNR: {enhanced_psnr}, SSIM: {enhanced_ssim}")

# Create subplots for PSNR and SSIM comparisons
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# PSNR comparison plot
ax1.bar(['Baseline', 'Enhanced'], [baseline_psnr, enhanced_psnr], color=['blue', 'green'])
ax1.set_title('PSNR Comparison')
ax1.set_ylabel('PSNR (dB)')

# SSIM comparison plot
ax2.bar(['Baseline', 'Enhanced'], [baseline_ssim, enhanced_ssim], color=['blue', 'green'])
ax2.set_title('SSIM Comparison')
ax2.set_ylabel('SSIM')

# Overall title
fig.suptitle('PSNR and SSIM Comparison Between Baseline and Enhanced Models')

# Save the figure
output_path = "C:/Users/USER/Desktop/Latest projects/deep_learning_project/results/psnr_ssim_comparison.png"
plt.savefig(output_path)
print(f"Comparison plot saved to {output_path}")

# Show the plot
plt.show()
