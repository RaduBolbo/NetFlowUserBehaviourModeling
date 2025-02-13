import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_stats(directory_path, specific_csv_path):
    # Initialize global histograms data
    global_packet_counts = []
    global_byte_counts = []

    # Iterate over all CSV files in the directory
    for filename in tqdm(os.listdir(directory_path)):
        file_path = os.path.join(directory_path, filename)
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            global_packet_counts.extend(df['packet_count'].tolist())
            global_byte_counts.extend(df['byte_count'].tolist())

    # Save global histograms
    plt.figure(figsize=(12, 6))

    print('1')

    plt.subplot(1, 2, 1)
    plt.hist(global_packet_counts, bins=50, color='blue', alpha=0.7)
    plt.title('Global Packet Count Histogram')
    plt.xlabel('Packet Count')
    plt.ylabel('Frequency')

    print('2')

    plt.subplot(1, 2, 2)
    plt.hist(global_byte_counts, bins=50, color='green', alpha=0.7)
    plt.title('Global Byte Count Histogram')
    plt.xlabel('Byte Count')
    plt.ylabel('Frequency')

    print('3')

    plt.tight_layout()
    plt.savefig('global_histograms_0.png')
    plt.close()

    # Load specific CSV for independent histogram
    specific_df = pd.read_csv(specific_csv_path)

    specific_packet_counts = specific_df['packet_count'].tolist()
    specific_byte_counts = specific_df['byte_count'].tolist()

    # Save specific histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(specific_packet_counts, bins=50, color='blue', alpha=0.7)
    plt.title(f'Specific Packet Count Histogram ({os.path.basename(specific_csv_path)})')
    plt.xlabel('Packet Count')
    plt.ylabel('Frequency')

    print('4')

    plt.subplot(1, 2, 2)
    plt.hist(specific_byte_counts, bins=50, color='green', alpha=0.7)
    plt.title(f'Specific Byte Count Histogram ({os.path.basename(specific_csv_path)})')
    plt.xlabel('Byte Count')
    plt.ylabel('Frequency')

    print('5')

    plt.tight_layout()
    plt.savefig('specific_histograms_0.png')
    plt.close()

    print('6')

    return {
        'global_packet_histogram': global_packet_counts,
        'global_byte_histogram': global_byte_counts,
        'specific_packet_histogram': specific_packet_counts,
        'specific_byte_histogram': specific_byte_counts
    }


if __name__ == '__main__':
    get_stats('dataset/train', 'dataset/train/output11.csv')