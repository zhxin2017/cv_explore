from torchvision.datasets import MNIST
from torchvision import transforms

train_mnist = MNIST(root="mnist", train=True)
# test_mnist = MNIST(root="mnist", train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
