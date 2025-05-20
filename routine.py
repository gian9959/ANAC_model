from anac_model.learning_functions import training
from anac_model.learning_functions import validation

starting_checkpoint = './checkpoints/0HiddenLayers/Epoch10_checkpoint.pth'
checkpoint_path = training(starting_checkpoint)
val_loss = validation(checkpoint_path)
