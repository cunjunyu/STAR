import matplotlib.pyplot as plt
import os
import argparse


def extract_encoder_type(log_file_name):
    """
    Extracts the encoder type from the log file name.
    
    Args:
        log_file_name (str): Log file name, which includes the encoder type.
    
    Returns:
        str: Extracted encoder type.
    """
    try:
        # Split the file name to extract the encoder type
        parts = log_file_name.split('_')
        encoder_type = parts[3]  # The 4th element in the split string is the encoder type
        return encoder_type
    except IndexError:
        raise ValueError("The log file name does not match the expected format.")


def plot_training_log(log_file_path, output_dir="plots"):
    """
    Reads the log file, parses the data, and plots train_loss and test_error.
    
    Args:
        log_file_path (str): Path to the log file.
        output_dir (str): Directory where the plot will be saved. Defaults to 'plots'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract encoder type from log file name
    log_file_name = os.path.basename(log_file_path)
    encoder_type = extract_encoder_type(log_file_name)

    # Initialize lists to hold parsed data
    epochs = []
    train_losses = []
    test_errors = []

    # Read and parse the log file
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                # Split the line by commas
                values = line.strip().split(',')
                if len(values) != 5:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                epoch, train_loss, test_error, _, _ = map(float, values)
                
                # Append data to respective lists
                epochs.append(int(epoch))
                train_losses.append(train_loss)
                test_errors.append(test_error)
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # Plot train_loss and test_error
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_errors, label='Test Error', marker='s')

    # Adding labels, legend, and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss/Error')
    plt.title(f"Training Curve: Encoder Type = {encoder_type}")
    plt.legend()
    plt.grid(True)

    # Save the plot
    num_epochs = len(epochs)
    plot_file_name = f"training_curve_{num_epochs}_epochs_{encoder_type}.png"
    plot_file_path = os.path.join(output_dir, plot_file_name)
    plt.savefig(plot_file_path, dpi=300)
    plt.close()
    print(f"Plot saved as: {plot_file_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot training curves from a log file.")
    parser.add_argument(
        "log_file_path", 
        type=str, 
        help="Path to the log file containing training data."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="plots",
        help="Directory where the plot will be saved. Defaults to 'plots'."
    )
    args = parser.parse_args()

    # Call the plotting function
    plot_training_log(args.log_file_path, args.output_dir)
