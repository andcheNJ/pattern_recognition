from scripts.model import SiameseNetwork
from scripts.loss import ContrastiveLoss
from scripts.utils import SiameseDataset
import torch
import os
from torch.utils.data import DataLoader, BatchSampler
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# main.py (top imports)
from scripts.utils import SiameseDataset, calculate_mean_std, visualize_embeddings
from collections import Counter

# --- Configuration ---
dataset_path = 'data'
epochs = 20
batch_size = 32
lr = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Calculate dataset statistics
dataset_mean, dataset_std = calculate_mean_std(dataset_path)


# --- Transforms ---
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize(mean=[dataset_mean], std=[dataset_std])
])

# --- Dataset & DataLoaders ---
dataset = SiameseDataset(dataset_path, transform=train_transform)

# Split dataset first
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



# Create balanced batch sampler for training
class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes=2, n_samples=16):
        self.labels = labels
        self.labels_set = list(set(labels))
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        for _ in range(len(self)):
            classes = random.sample(self.labels_set, self.n_classes)
            indices = []
            for cls in classes:
                cls_indices = [i for i, l in enumerate(self.labels) if l == cls]
                
                # FIX: Handle case where class has < n_samples
                available_samples = min(len(cls_indices), self.n_samples)
                selected = random.sample(cls_indices, available_samples)
                
                # If we need more samples, fill with random choices
                if len(selected) < self.n_samples:
                    selected += random.choices(cls_indices, 
                                             k=self.n_samples - len(selected))
                
                indices.extend(selected)
            random.shuffle(indices)
            yield indices

    def __len__(self):
        return len(self.labels) // self.batch_size

# FIX 1: Get labels correctly from the subset
train_labels = [dataset.labels[i] for i in train_dataset.indices]
train_sampler = BalancedBatchSampler(
    train_labels,
    n_classes=4,  # Reduce if you have few classes
    n_samples=8   # Reduce if some classes have <16 samples
)
print("Samples per class:", Counter(train_labels))
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Model Setup ---
model = SiameseNetwork()
criterion = ContrastiveLoss()
# Change learning rate and add weight decay
optimizer = torch.optim.Adam(model.parameters(), 
                           lr=0.0001,  # Reduced from 0.001
                           weight_decay=1e-5)  # Regularization
#torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Replace current scheduler with cosine annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                     T_max=epochs//2, 
                                                     eta_min=1e-6)


model = SiameseNetwork().to(device)

# --- Training Loop ---
best_val_loss = float('inf')
patience_counter = 0  # Add this for early stopping

for epoch in range(epochs):
    # Training Phase
    model.train()
    train_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # FIX 2: Convert to float32 and check device
        images = images.float().to(device)
        labels = labels.to(device)
        
        # Create pairs
        pairs = []
        pair_labels = []
        
        for i in range(len(images)):
            # Positive pair
            same_class = torch.where(labels == labels[i])[0]
            if len(same_class) > 1:  # Ensure there's at least 2 samples
                j = random.choice(same_class[same_class != i])
                pairs.append((images[i], images[j]))
                pair_labels.append(1)
            
            # Negative pair
            diff_class = torch.where(labels != labels[i])[0]
            if len(diff_class) > 0:
                j = random.choice(diff_class)
                pairs.append((images[i], images[j]))
                pair_labels.append(0)
        
        if not pairs:  # Skip batches with no valid pairs
            continue
            
        # Convert to tensors
        input1 = torch.stack([p[0] for p in pairs])
        input2 = torch.stack([p[1] for p in pairs])
        pair_labels = torch.tensor(pair_labels, device=labels.device)
        
        # Forward pass
        output1, output2 = model(input1, input2)
        loss = criterion(output1, output2, pair_labels)
        
        # Backward passo
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
        
        # FIX 3: Moved print inside batch loop
        print(f"Processed {batch_idx+1}/{len(train_loader)} batches", end='\r')
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            # FIX 4: Add validation pair creation
            images = images.float().to(device)
            labels = labels.to(device)
            
            # Create random pairs for validation
            rand_indices = torch.randperm(len(images))
            input1 = images
            input2 = images[rand_indices]
            pair_labels = (labels == labels[rand_indices]).float()
            
            output1, output2 = model(input1, input2)
            loss = criterion(output1, output2, pair_labels)
            val_loss += loss.item()
    
    # Calculate metrics
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    # Scheduler step
    scheduler.step(avg_val_loss)
    
    # Print progress
    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print(f"Positive pairs: {(pair_labels == 1).sum().item()}")
    print(f"Negative pairs: {(pair_labels == 0).sum().item()}")
    
    # Save best model
    # After validation phase
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= 5:
            print("Early stopping triggered")
            break
        
        if epoch % 5 == 0:  # Only visualize every 5 epochs
            visualize_embeddings(model, val_loader)

# Final save
torch.save(model.state_dict(), r'C:\Users\Andrew\OneDrive\Dokumente\Arduino\image_project\models\model_1.pth')