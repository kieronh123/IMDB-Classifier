import matplotlib.pyplot as plt

# Replace with real values from your logs
epochs = [1, 2]
accuracy = [0.84, 0.88]

plt.plot(epochs, accuracy, marker='o')
plt.title("Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("assets/accuracy_plot.png")
plt.show()
