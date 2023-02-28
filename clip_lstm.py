# 1. install the necessary libraries:

#pip install torch torchvision transformers


# 2. Import the required modules:

import torch
import torchvision
from transformers import CLIP, AdamW


#Define your custom image and text encoders. 
# In this example, we will use ResNet50 for image encoding and a simple LSTM for text encoding. 
# These encoders will be passed to the CLIP model later.

class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        return x

class TextEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_size)
        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]

        return x

# 4. Load your dataset. In this example, we will use the CIFAR10 dataset as an example.

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)


# 5. Initialize your model with the custom encoders. 
# Here we will initialize our CLIP model with our custom image and text encoders.

image_encoder = ImageEncoder()
text_encoder = TextEncoder(vocab_size=10000, embedding_size=128, hidden_size=256)

model = CLIP(visual_encoder=image_encoder, text_encoder=text_encoder)

# 6. Define your loss function and optimizer. In this example, 
# we will use the contrastive loss function and AdamW optimizer.


criterion = torch.nn.CosineEmbeddingLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)

# 7. Train your model. In this example, we will train the model for 10 epochs.

for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        images = inputs.to(device)
        captions = labels.to(device)

        optimizer.zero_grad()

        # Get image and text embeddings
        image_embeddings = model.encode_image(images)
        text_embeddings = model.encode_text(captions)

        # Compute contrastive loss
        targets = torch.ones(images.shape[0]).to(device)
        loss = criterion(image_embeddings, text_embeddings, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print statistics every 1000 batches
        if i % 1000 == 999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

    # Evaluate model on validation set and calculate F1 score
    with torch.no_grad():
        model.eval()
        f1_score = 0.0
        for data in valloader:
            inputs, labels = data
            images = inputs.to(device)
            captions = labels.to(device)

            # Get image and text embeddings
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(captions)

            # Compute cosine similarity between embeddings
            similarities = torch.nn.functional.cosine_similarity(image_embeddings, text_embeddings)

            # Calculate F1 score
            predictions = (similarities > 0.5).float()
            tp = (predictions * captions.float()).sum().item()
            fp = ((1 - predictions) * captions.float()).sum().item()
            fn = (predictions * (1 - captions.float())).sum().item()

            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1_score += 2 * precision * recall / (precision + recall + 1e-10)

        f1_score /= len(valloader)

        print('Epoch %d, F1 score: %.3f' % (epoch + 1, f1_score))
        model.train()

            
        ##################################### here we add and calculate F1 score with the following code######################
        

