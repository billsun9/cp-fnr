import torch

from utils import *
from train import *
from models.baseline_mlp import DeepMultiLabelClassifier

if __name__ == "__main__":
    X_train, Y_train, X_calibration, Y_calibration, X_test, Y_test = load_data_splits()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device {}".format(device))
    # Parameters
    input_size = len(X_train[0])
    output_size = len(Y_train[0])
    hidden_size = 256
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.000005
    
    print(f"Input Size: {input_size}")
    print(f"Output Size: {output_size}")
    print(f"Hidden Size: {hidden_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    
    # Instantiate model and load data
    model = DeepMultiLabelClassifier(input_size, hidden_size, output_size).to(device)
    train_loader = load_data(X_train, Y_train, device, batch_size)
    val_loader = load_data(X_calibration, Y_calibration, device, batch_size)
    test_loader = load_data(X_test, Y_test, device, batch_size)
    print(model)
    # Train the model
    hist = train_model(model, train_loader, val_loader, num_epochs=num_epochs, learning_rate=learning_rate)

    print("\t[Evaluation on training data...]")
    pred_raw = model(torch.tensor(X_train, dtype=torch.float32).to(device))
    pred = torch.round(pred_raw)
    eval_metrics(Y_train, pred.to("cpu").detach().numpy())

    print("\t[Evaluation on calibration data...]")
    pred_raw = model(torch.tensor(X_calibration, dtype=torch.float32).to(device))
    pred = torch.round(pred_raw)
    eval_metrics(Y_calibration, pred.to("cpu").detach().numpy())
    
    print("\t[Evaluation on testing data...]")
    pred_raw = model(torch.tensor(X_test, dtype=torch.float32).to(device))
    pred = torch.round(pred_raw)
    eval_metrics(Y_test, pred.to("cpu").detach().numpy())
    
    print("Done")