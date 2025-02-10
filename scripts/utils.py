from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.utils.data import DataLoader  # Add missing import

class SiameseDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        
        # List ONLY directories (ignore files)
        self.classes = [d.name for d in os.scandir(dataset_path) if d.is_dir()]
        
        self.image_paths = []
        self.labels = []
        
        # Create mapping from class name to numeric label
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load images from each class directory
        for class_name in self.classes:
            class_path = os.path.join(dataset_path, class_name)
            if not os.path.isdir(class_path):
                continue  # Skip non-directory items
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path):  # Only add actual files
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    # ADD THIS METHOD
    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.image_paths)

    def __getitem__(self, index):
        try:
            img_path = self.image_paths[index]
            img = Image.open(img_path).convert('L')  # Grayscale
            if self.transform:
                img = self.transform(img)
            return img, self.labels[index]
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None
        
def calculate_mean_std(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    
    dataset = SiameseDataset(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, -1)
        mean += images.mean(1).sum()
        std += images.std(1).sum()
    
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.item(), std.item()

def visualize_embeddings(model, dataloader):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            outputs = model.base_network(images)
            embeddings.append(outputs)
            labels.append(batch_labels)
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, perplexity=15)
    reduced = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap='tab10')
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
                
        