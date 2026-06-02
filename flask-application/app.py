#Importing all libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from PIL import Image
from collections import Counter

#building the base model using imagenet pre-trained weights and replacing the final layer to fit the 3 classes
def bmodel(num_classes=3, freeze_base=True):
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    feat = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(feat, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model

def train_epoch(model, loader, optimizer, criterion, gpu):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(gpu), labels.to(gpu)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total

def val_epoch(model, loader, criterion, gpu):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(gpu), labels.to(gpu)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def extract_test(img, transform, tile_size=512, overlap=64):
    w, h = img.size
    step = tile_size - overlap
    test_images, positions = [], []
    for top in range(0, h - tile_size + 1, step):
        for left in range(0, w - tile_size + 1, step):
            box = (left, top, left + tile_size, top + tile_size)
            tile = img.crop(box)
            test_images.append(transform(tile))
            positions.append(box)
    return torch.stack(test_images), positions


def predicting_test(image_path, model, transform, mangrove_type, gpu, batch_size=64):
    img = Image.open(image_path).convert('RGB')
    tiles, _ = extract_test(img, transform, 512, 64)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size].to(gpu)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs  = np.concatenate(all_probs, axis=0)
    avg_probs  = np.mean(all_probs, axis=0)
    pred_idx   = np.argmax(avg_probs)
    confidence = avg_probs[pred_idx]
    return mangrove_type[pred_idx], float(confidence), avg_probs

if __name__ == '__main__':

    #configurations before running any primary code
    dir = r'C:\Users\FSOS\Music\split_mangrove_images' #WE NEED TO CHANGE THiS FOR THE MAIN REPO
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #using my 3070 for training, will most likely have to edit this for cpu use in main repo
    testing_dir    = r'C:\Users\FSOS\Music\internship\test'
    mangrove_type = ['orange', 'red', 'yellow']

    #Loading images (they have already been pre-processed and split into smaller images)
    image_dataset = datasets.ImageFolder(root=dir)
    mangrove_type  = image_dataset.classes

    #splitting the dataset into a 80/20 train/validation split
    total = len(image_dataset)
    training = int(0.8 * total)
    validating   = total - training
    train_set, val_set = torch.utils.data.random_split(image_dataset, [training, validating],generator=torch.Generator().manual_seed(42))

    #Creating and using transformation for our dataset, using the standard deviation and mean from imagenet for the normailization
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    train_set.dataset.transform = train_transform
    val_set.dataset.transform   = val_transform
    print(f"Total train set size: {len(train_set)} | Total validation set size: {len(val_set)}")

    #Couting the number of images in each class for the training set to handle class imbalance with a weighted sampler
    training_label = [image_dataset.targets[i] for i in train_set.indices]
    counting_class = np.bincount(training_label)
    class_weights = 1.0 / counting_class
    weights = [class_weights[l] for l in training_label]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = bmodel(freeze_base=True).to(gpu)

    #training and validation using epoch and predefining varaibles for the epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    val_cc = 0
    pat_counter = 0


    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, gpu)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, gpu)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:02d}/{50} | "f"training loss: {train_loss:.4f} accuracy: {train_acc:.4f} | "f"validation loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

        if val_acc > val_cc:
            val_cc = val_acc
            torch.save(model.state_dict(), 'best_mangrove_model.pth')
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= 5:
                print("Now Early stopping")
                break

    #Model Layer 2
    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.features.parameters())[-20:]:
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    pat_counter = 0

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, gpu)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, gpu)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:02d}/{50} | "f"training loss: {train_loss:.4f} accuracy: {train_acc:.4f} | "f"validation loss: {val_loss:.4f} accuracy: {val_acc:.4f}")
        if val_acc > val_cc:
            val_cc = val_acc
            torch.save(model.state_dict(), 'best_mangrove_model.pth')
            pat_counter = 0
        else:
            pat_counter += 1
            if pat_counter >= 5:
                print("Now Early stopping")
                break

    #Model evaluation using the training, testing and validation set
    print("\nEvaluating on validation datset")
    model.load_state_dict(torch.load('best_mangrove_model.pth'))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(gpu)
            outputs = model(images)
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, target_names=mangrove_type))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=mangrove_type, yticklabels=mangrove_type)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Mangrove Classification')
    plt.tight_layout()
    plt.show()

    #Reading and the test image and predicting on the test images


    #Evaluation and classifcation report
    file_formats = ('.jpg', '.jpeg', '.png', '.tif', '.tiff') # Update if you have other formats
    y_true, y_pred = [], []
    results = []

    for true_class in mangrove_type:
        class_dir = os.path.join(testing_dir, true_class)
        if not os.path.exists(class_dir):
            print(f"Skipping {true_class} folder not found")
            continue
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(file_formats)]
        for fname in images:
            path = os.path.join(class_dir, fname)
            pred_class, confidence, avg_probs = predicting_test(path)
            y_true.append(mangrove_type.index(true_class))
            y_pred.append(mangrove_type.index(pred_class))
            results.append({
                'file':       fname,
                'true':       true_class,
                'predicted':  pred_class,
                'confidence': confidence,
                'correct':    pred_class == true_class,
                'probs':      avg_probs
            })

    print(classification_report(y_true, y_pred, target_names=mangrove_type))


    print("Mangrove Type Accuracy:")
    for cls in mangrove_type:
        cls_results = [r for r in results if r['true'] == cls]
        if cls_results:
            correct = sum(r['correct'] for r in cls_results)
            print(f"  {cls:<8} {correct}/{len(cls_results)} ({correct/len(cls_results)*100:.1f}%)")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=mangrove_type, yticklabels=mangrove_type,cmap='Greens')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Testing Set')
    plt.tight_layout()
    plt.show()