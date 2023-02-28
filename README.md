# medical_CLIP
This medical Contrastive Language-Image Pre-Training will be used on MIMIC dataset to predict classes

# lstm_clip

1. Define our custom image and text encoders. In this example, we will use ResNet50 for image encoding and a simple LSTM for text encoding. These encoders will be passed to the CLIP model later.

2. Load our dataset. In this example, we will use the CIFAR10 dataset as an example.
3. Initialize our model with the custom encoders. Here we will initialize our CLIP model with our custom image and text encoders.
4. Define our loss function and optimizer. In this example, we will use the contrastive loss function and AdamW optimizer.
5. Train our model. In this example, we will train the model for 10 epochs.


6. In the training code , we first evaluate the model on the validation set and calculate the F1 score. To do this, we set the model to evaluation mode using `model.eval()`, then we loop over the validation set and calculate the cosine similarity between the image and text embeddings. We then threshold the similarities at 0.5 to obtain binary predictions, and compute the F1 score using these predictions and the ground truth labels. Finally, we set the model back to training mode using `model.train()`. We repeat this evaluation and F1 score calculation after each epoch of training.

# Roberta_clip_optuna

To use the Roberta model for text encoding and add a hyperparameter search using Optuna, we proceeded as follow:

1. Define our custom image and text encoders. In this example, we will use ResNet50 for image encoding and Roberta for text encoding. These encoders will be passed to the CLIP model later

2. Load our dataset. In this example, we will use the CIFAR10 dataset as an example.

3. Initialize your model with the custom encoders. Here we will initialize our CLIP model with our custom image and text encoders.

4. Define your loss function and optimizer. In this example, we will use the contrastive loss function and AdamW optimizer.

5. Define a function to train the model with Optuna hyperparameter search. We will define the number of epochs, learning rate, and batch size as hyperparameters to search over.

6. Inside the training loop, we will train the model using the current hyperparameters.

7. At the end of each epoch, we will compute the validation loss and return it as the objective value for Optuna to minimize.

8. Run the Optuna hyperparameter search.

9. Get the best hyperparameters found by Optuna and retrain the model using those hyperparameters.

10. Using a function, we calculated the  `F1 score` on the test set by comparing the binary predictions to the ground truth labels using the `compute_f1_score` function. Note that we need to convert the predictions and labels back to CPU before computing the `F1 score`.