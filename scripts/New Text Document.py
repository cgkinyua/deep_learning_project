import matplotlib.pyplot as plt

# Example data from the iterative refinement results
epochs = list(range(1, 21))
train_loss = [0.0276, 0.0108, 0.0075, 0.0064, 0.0058, 0.0055, 0.0053, 0.0052, 0.0051, 0.0050,
              0.0048, 0.0048, 0.0048, 0.0047, 0.0047, 0.0046, 0.0046, 0.0046, 0.0045, 0.0045]
val_loss = [0.0139, 0.0087, 0.0068, 0.0060, 0.0057, 0.0055, 0.0053, 0.0052, 0.0051, 0.0050,
            0.0049, 0.0048, 0.0048, 0.0047, 0.0047, 0.0047, 0.0046, 0.0046, 0.0046, 0.0046]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label="Training Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs. Validation Loss (Iterative Refinement Model)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("C:/Users/USER/Desktop/Latest projects/deep_learning_project/results/training_validation_loss_refinement.png")
plt.show()
