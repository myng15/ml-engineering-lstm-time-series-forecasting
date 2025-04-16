import argparse
import mlflow
import torch
from torch.utils.data import DataLoader, TensorDataset
from app.model import LSTMPredictor
from app.utils import *
from train_utils import train_model
import matplotlib.pyplot as plt
import os
import numpy as np

DATA_PATH = "data/time_series_19-covid-Confirmed_archived_0325.csv"
daily_cases = load_and_process_data(DATA_PATH)
normalized_data, scaler = normalize_data(daily_cases)

def main(args):
    seed_everything()

    # ----------------------------- Setup -----------------------------

    # Prepare data
    if args.finalize_model:
        X_train, y_train = create_sequences(normalized_data, args.seq_length)
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test, y_test = None, None
    else:
        X_train, y_train, X_test, y_test = prepare_data(
            dataset_path=DATA_PATH, 
            seq_length=args.seq_length, 
            test_size=args.test_size,
        )

    # Data loaders
    # train_dataset = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_dataset = TensorDataset(X_test, y_test)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Instantiate model
    model = LSTMPredictor(
        input_size=1,
        hidden_size=args.hidden_size,
        seq_len=args.seq_length,
        n_layers=args.n_layers,
        dropout=args.dropout
    )

    # ----------------------------- Training -----------------------------

    # Train model
    model, train_losses, test_losses = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=torch.device(args.device)
    )

    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    if not args.finalize_model:
        plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if args.finalize_model:
        plt.title("Finalized Model's Training Loss")
        #plt.ylim((0, 5))
        plot_path = "plots/loss_curve_finalized_model.png"
        model_path = "models/finalized_lstm.pt"
    else:
        plt.title("Training vs Test Loss")
        #plt.ylim((0, 5))
        plot_path = "plots/loss_curve.png"
        model_path = "models/trained_lstm.pt"
    
    plt.savefig(plot_path)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
        
    print("Training complete.")


    # ----------------------------- Inference (Predicting test data) -----------------------------
    if not args.finalize_model:
        test_seq = X_test[:1] #initial test seq
        preds = []

        with torch.no_grad():
            for _ in range(len(X_test)):
                y_pred = model(test_seq)
                pred = torch.flatten(y_pred).item()
                preds.append(pred)

                new_seq = test_seq.numpy().flatten()
                new_seq = np.append(new_seq, [pred])
                new_seq = new_seq[1:]
                test_seq = torch.tensor(new_seq).view(1, args.seq_length, 1).float()

        true_cases = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
        predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

        split_idx = int(len(normalized_data) * (1 - args.test_size))
        train_data = normalized_data[:split_idx]

        plt.figure()
        plt.plot(
            daily_cases.index[:len(train_data)],
            scaler.inverse_transform(train_data).flatten(),
            label="Historical Daily Cases"
        )
        plt.plot(
            daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
            true_cases,
            label="Real Daily Cases"
        )
        plt.plot(
            daily_cases.index[len(train_data):len(train_data) + len(true_cases)],
            predicted_cases,
            label="Predicted Daily Cases"
        )
        plt.legend()
        test_plot_path = os.path.join("plots/historical_real_predicted_curve.png")
        plt.savefig(test_plot_path)
        print("Plot saved at ", test_plot_path)

        print("Inference (predicting test data) complete.")


    # ----------------------------- Inference (Predicting future) -----------------------------
    if args.finalize_model:
        n_days = 12
        X_train, _ = create_sequences(normalized_data, args.seq_length)
        X_train = torch.tensor(X_train).float()

        #test_seq = X_train[:1] # Use the first historical sequence as initial test seq
        test_seq = X_train[-1:] # Use the most recent historical sequence as initial test seq
        preds = []

        with torch.no_grad():
            for _ in range(n_days):
                y_pred = model(test_seq)
                pred = torch.flatten(y_pred).item()
                preds.append(pred)

                new_seq = test_seq.numpy().flatten()
                new_seq = np.append(new_seq, [pred])
                new_seq = new_seq[1:]
                test_seq = torch.tensor(new_seq).view(1, args.seq_length, 1).float()

        predicted_cases = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()

        predicted_index = pd.date_range(
            start=daily_cases.index[-1] + pd.Timedelta(days=1),
            periods=n_days,
        )
        predicted_series = pd.Series(data=predicted_cases, index=predicted_index)
        print("predicted_series: ", predicted_series)

        # # Save plots
        plt.figure()
        plt.plot(predicted_series, label="Predicted Daily Cases")
        plt.legend()
        future_predicted_plot_path = os.path.join("plots/predicted_curve_finalized_model.png")
        plt.savefig(future_predicted_plot_path)
        print("Plot saved at ", future_predicted_plot_path)

        plt.figure()
        plt.plot(daily_cases, label="Historical Daily Cases")
        plt.plot(predicted_series, label="Predicted Daily Cases")
        plt.legend()
        future_historical_predicted_plot_path = os.path.join("plots/historical_predicted_curve_finalized_model.png")
        plt.savefig(future_historical_predicted_plot_path)
        print("Plot saved at ", future_historical_predicted_plot_path)

        print("Inference (predicting future) complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, default="LSTM-TimeSeries")
    parser.add_argument("--finalize_model", action='store_true', help="Use all available data for training (i.e. no train-test split)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seq_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    main(args)
