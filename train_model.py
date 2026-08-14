import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import collections

# ==================== CONFIGURATION ====================
DATASET_PATH = './devanagari_dataset'
MODEL_SAVE_PATH = './models/devanagari_ocr_model.pth'
EPOCHS = 5
BATCH_SIZE = 16  # Reduced for stability
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Character name to code mapping
CHAR_MAP = {
    # Vowels
    'a': 'अ', 'aa': 'आ', 'i': 'इ', 'ii': 'ई', 'u': 'उ', 'uu': 'ऊ',
    'ri': 'ऋ', 'ri_i': 'ऌ', 'e': 'ए', 'ai': 'ऐ', 'o': 'ओ', 'au': 'औ',
    
    # Consonants
    'ka': 'क', 'kha': 'ख', 'ga': 'ग', 'gha': 'घ', 'kna': 'ङ',
    'cha': 'च', 'chha': 'छ', 'ja': 'ज', 'jha': 'झ', 'nya': 'ञ',
    'ta': 'ट', 'tha': 'ठ', 'da': 'ड', 'dha': 'ढ', 'na': 'ण',
    't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
    'pa': 'प', 'pha': 'फ', 'ba': 'ब', 'bha': 'भ', 'ma': 'म',
    'ya': 'य', 'yaw': 'य', 'ra': 'र', 'la': 'ल', 'va': 'व',
    'sha': 'श', 'ssa': 'ष', 'sa': 'स', 'ha': 'ह', 'lla': 'ळ',
    
    # Additional/Variant characters (map to main characters)
    'yna': 'य', 'waw': 'व', 'taamatar': 'त', 'thaa': 'थ', 'daa': 'द',
    'dhaa': 'ध', 'adna': 'ण', 'tabala': 'त', 'motosaw': 'स',
    'petchiryakha': 'छ', 'patalosaw': 'स', 'chhya': 'च', 'tra': 'त', 'gya': 'ज',
    
    # Digits
    '0': '०', '1': '१', '2': '२', '3': '३', '4': '४',
    '5': '५', '6': '६', '7': '७', '8': '८', '9': '९',
}

# ==================== DATASET CLASS ====================
class DevanagariDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # First pass: collect all data
        self._collect_data()
        
        # Create label to index mapping based on unique characters found
        unique_chars = sorted(set(self.labels))
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.num_classes = len(unique_chars)
        
        # Convert character labels to indices
        self.label_indices = [self.char_to_idx[label] for label in self.labels]
        
        print(f"\n=== CHARACTER SET ===")
        print(f"Total unique characters found: {self.num_classes}")
        print(f"Characters: {unique_chars}")
        print(f"Total images: {len(self.images)}")
    
    def _collect_data(self):
        """Collect all data in first pass"""
        images_dir = self.dataset_path / 'Images'
        
        if not images_dir.exists():
            print(f"ERROR: Images directory not found at {images_dir}")
            return
        
        # Get all character folders
        character_folders = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        print(f"\nScanning {len(character_folders)} character folders...")
        
        loaded_count = 0
        skipped_folders = []
        
        for char_folder in character_folders:
            # Extract character from folder name
            char = self._get_character(char_folder.name)
            
            if char is None:
                skipped_folders.append(char_folder.name)
                continue
            
            # Get all PNG files
            png_files = list(char_folder.glob('*.png'))
            
            if len(png_files) == 0:
                skipped_folders.append(f"{char_folder.name} (no images)")
                continue
            
            # Add to dataset
            for img_file in png_files:
                self.images.append(str(img_file))
                self.labels.append(char)
            
            print(f"   {char_folder.name} ({char}): {len(png_files)} images")
            loaded_count += 1
        
        if skipped_folders:
            print(f"\n⚠ Skipped {len(skipped_folders)} folders:")
            for folder in skipped_folders[:10]:
                print(f"  - {folder}")
            if len(skipped_folders) > 10:
                print(f"  ... and {len(skipped_folders) - 10} more")
    
    def _get_character(self, folder_name):
        """Extract character from folder name"""
        # Try to get the last part
        parts = folder_name.split('_')
        if len(parts) >= 2:
            char_code = parts[-1].lower()
            return CHAR_MAP.get(char_code)
        
        # Try just the name
        return CHAR_MAP.get(folder_name.lower())
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        try:
            img = Image.open(img_path).convert('L')  # Grayscale
        except:
            # If image can't be opened, return a blank image
            img = Image.new('L', (32, 32))
        
        if self.transform:
            img = self.transform(img)
        
        label = self.label_indices[idx]
        
        return img, label

# ==================== MODEL ====================
class DevanagariOCRModel(nn.Module):
    def __init__(self, num_classes):
        super(DevanagariOCRModel, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== TRAINING ====================
def train_model():
    """Train the model"""
    
    # Create output directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Load dataset
    print("\n" + "="*70)
    print("LOADING DATASET")
    print("="*70)
    dataset = DevanagariDataset(DATASET_PATH, transform=transform)
    
    if len(dataset) == 0:
        print("\nERROR: No images loaded!")
        return
    
    num_classes = dataset.num_classes
    print(f"\n Successfully loaded {len(dataset)} images")
    print(f" Number of character classes: {num_classes}")
    
    # Split dataset
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f" Training set: {train_size} images")
    print(f" Validation set: {val_size} images")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0
    )
    
    # Model with correct number of classes
    device = torch.device(DEVICE)
    model = DevanagariOCRModel(num_classes=num_classes).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    print(f"\n" + "="*70)
    print(f"TRAINING ON {str(device).upper()}")
    print("="*70 + "\n")
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{EPOCHS}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'   Model saved! Accuracy: {val_acc:.2f}%')
    
    print(f"\n" + "="*70)
    print(f"TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Number of character classes: {num_classes}")
    print("="*70)

if __name__ == '__main__':
    train_model()