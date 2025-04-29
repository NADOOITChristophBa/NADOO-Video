# Dataset Catalog for Dynamic Parameter Experiments

This catalogue lists datasets to showcase adaptive model-parameter methods (chunked routing, dynamic quantization, boundary pruning). Use these in your benchmark scripts.

## 1. Synthetic Gaussian Classification
- **Task:** Binary or multi-class on random vectors.
- **Dimensions:** configurable `D` (e.g. 128, 256).
- **Loader Example:**
```python
from nadoo_algorithmen.datasets import synthetic_gaussian_loader

# Create a synthetic data loader
loader = synthetic_gaussian_loader(
    n_samples=10000,
    D=128,
    n_classes=2,
    batch_size=64,
    shuffle=True
)
# Iterate batches
for X, y in loader:
    # X: [batch_size, D], y: labels
    ...
```

## 2. MNIST (Handwritten digits)
- **Description:** 28×28 grayscale images, 10 classes.
- **Loader:**
```python
from nadoo_algorithmen.datasets import get_loaders

train_loader, test_loader = get_loaders(
    name='mnist',
    batch_size=64,
    data_dir='./data'
)
```

## 3. FashionMNIST
- Same as MNIST but fashion items (10 classes).
- Replace `datasets.MNIST` with `datasets.FashionMNIST`.

## 4. CIFAR-10 (Tiny color images)
- **Shape:** 32×32×3, 10 classes.
- **Loader:**
```python
from torchvision import datasets, transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])
train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test  = datasets.CIFAR10('./data', train=False, transform=transform)
```

## 5. CIFAR-100
- **Shape:** 32×32×3, 100 classes.
- Loader is identical to CIFAR-10 but with `CIFAR100`.

## 6. SVHN
- Street View House Numbers dataset.
- Use `datasets.SVHN` with `split='train'` and `split='test'`.

## 7. Tiny ImageNet
- **Shape:** 64×64×3, 200 classes.
- Download from [TinyImageNet](http://tiny-imagenet.herokuapp.com/).
- Custom `ImageFolder` loader:
```python
from torchvision import transforms, datasets
transform = transforms.Compose([...])
data = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform)
loader = DataLoader(data, batch_size=64, shuffle=True)
```

## 8. Custom Synthetic Shapes
- Dynamically generate inputs of varying shapes (e.g., sequence length, image resolution) to stress-test dynamic loading.
- Example: variable-length token sequences or random image sizes.

---

Use these datasets to compare baseline vs. adaptive methods. You can mix & match in your scripts via a common loader factory.
