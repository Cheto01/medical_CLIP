#1. Install the necessary libraries

#pip install torch torchvision transformers optuna

# 2. Import the required modules
import torch
import torchvision
from transformers import CLIP, AdamW, RobertaModel, RobertaTokenizer
import optuna


# 3.  Define your custom image and text encoders. In this example, 
# we will use ResNet50 for image encoding and Roberta for text encoding. 
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
    def __init__(self, model_name):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)

    def forward(self, x):
        input_ids = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')['input_ids'].to(device)
        output = self.roberta(input_ids)[0][:, 0, :]
        return output


# 4  Load your dataset. In this example, we will use the CIFAR10 dataset as an example.

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

# 5. Initialize your model with the custom encoders. 
# Here we will initialize our CLIP model with our custom image and text encoders.


image_encoder = ImageEncoder()
text_encoder = TextEncoder('roberta-base')

model = CLIP(visual_encoder=image_encoder, text_encoder=text_encoder)

# 6. Define your loss function and optimizer. 
# In this example, we will use the contrastive loss function and AdamW optimizer.

criterion = torch.nn.CosineEmbeddingLoss()
optimizer = AdamW(model.parameters(), lr=0.0001)


# 7. Define a function to train the model with Optuna hyperparameter search. 
# We will define the number of epochs, learning rate, and batch size as hyperparameters to search over.

def objective(trial):
    # Define hyperparameters to search over
    num_epochs = 10
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0

        # 8. Inside the training loop, we will train the model using the current hyperparameters.
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Compute image and text embeddings
            image_embeddings = model.get_image_features(inputs)
            text_embeddings = model.get_text_features(labels)

            # Compute contrastive loss
            targets = torch.ones(batch_size, dtype=torch.float).to(device)
            loss = criterion(image_embeddings, text_embeddings, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # Print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                
        # 9. At the end of each epoch, we will compute the validation loss and return 
        # it as the objective value for Optuna to minimize.
        # Compute validation loss
        validation_loss = 0.0
        with torch.no_grad():
            for data in validationloader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Compute image and text embeddings
                image_embeddings = model.get_image_features(inputs)
                text_embeddings = model.get_text_features(labels)

                # Compute contrastive loss
                targets = torch.ones(batch_size, dtype=torch.float).to(device)
                loss = criterion(image_embeddings, text_embeddings, targets)

                validation_loss += loss.item()

        # Return validation loss as objective value for Optuna to minimize
        return validation_loss

# 10. Run the Optuna hyperparameter search.

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# 11. Get the best hyperparameters found by Optuna and retrain the model using those hyperparameters.

# Get best hyperparameters found by Optuna
best_params = study.best_params

# Retrain the model using the best hyperparameters
num_epochs = 10
learning_rate = best_params['learning_rate']
batch_size = best_params['batch_size']

optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Compute image and text embeddings
        image_embeddings = model.get_image_features(inputs)
        text_embeddings = model.get_text_features(labels)

        # Compute contrastive loss
        targets = torch.ones(batch_size, dtype=torch.float).to(device)
        loss = criterion(image_embeddings, text_embeddings, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0


##################################### here we add and calculate F1 score with the following code######################

# Define function to compute F1 score
def compute_f1_score(labels, predictions):
    true_positives = (labels * predictions).sum().item()
    false_positives = ((1 - labels) * predictions).sum().item()
    false_negatives = (labels * (1 - predictions)).sum().item()
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    return f1_score

# Evaluate the model on the test set
with torch.no_grad():
    all_predictions = []
    all_labels = []

    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Compute image and text embeddings
        image_embeddings = model.get_image_features(inputs)
        text_embeddings = model.get_text_features(labels)

        # Compute cosine similarity between image and text embeddings
        similarities = F.cosine_similarity(image_embeddings, text_embeddings)

        # Convert similarities to binary predictions
        predictions = (similarities > 0.5).float()

        all_predictions.append(predictions)
        all_labels.append(labels)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # Compute accuracy and F1 score
    accuracy = (all_predictions == all_labels).float().mean().item()
    f1_score = compute_f1_score(all_labels.cpu(), all_predictions.cpu())

    print('Accuracy on test set: %.3f' % accuracy)
    print('F1 score on test set: %.3f' % f1_score)


