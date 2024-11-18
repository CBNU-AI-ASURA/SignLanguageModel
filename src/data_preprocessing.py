# data_preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, pad_sequences
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def find_keypoints_in_file(file):
    keypoints = set()
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        keypoints.add((row['landmark_type'], row['index']))
    return keypoints

def get_unique_keypoints(data_dir):
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(find_keypoints_in_file, csv_files), total=len(csv_files), desc="Finding unique keypoints"))
    
    unique_keypoints = set().union(*results)
    sorted_keypoints = sorted(list(unique_keypoints))
    return sorted_keypoints

def split_csv_into_segments(df):
    segments = []
    current_segment = []
    current_label = None
    df_sorted = df.sort_values(by='time').reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        label = row['label']
        if current_label is None:
            current_label = label
            current_segment.append(row)
        elif label == current_label:
            current_segment.append(row)
        else:
            segments.append((current_label, pd.DataFrame(current_segment)))
            current_label = label
            current_segment = [row]
    if current_segment:
        segments.append((current_label, pd.DataFrame(current_segment)))
    return segments

def process_file(args):
    file, sorted_keypoints = args
    sequences = []
    labels = []
    df = pd.read_csv(file)
    segments = split_csv_into_segments(df)

    for label, segment_df in segments:
        frame_data = []
        grouped = segment_df.groupby('frame_num')
        for frame_num, group in grouped:
            frame_keypoints = {}
            for _, row in group.iterrows():
                key = (row['landmark_type'], row['index'])
                frame_keypoints[key] = [row['x'], row['y'], row['z']]
            
            frame_vector = []
            for key in sorted_keypoints:
                if key in frame_keypoints:
                    frame_vector.extend(frame_keypoints[key])
                else:
                    frame_vector.extend([0.0, 0.0, 0.0])
            frame_data.append(frame_vector)
        
        sequences.append(frame_data)
        labels.append(label)
    
    return sequences, labels

def load_data(data_dir):
    sorted_keypoints = get_unique_keypoints(data_dir)
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_file, [(file, sorted_keypoints) for file in csv_files]), total=len(csv_files), desc="Processing files"))

    sequences, labels = zip(*results)
    sequences = [seq for batch in sequences for seq in batch]
    labels = [label for batch in labels for label in batch]

    return pad_sequences(sequences, dtype='float32', padding='post', value=0.0), labels

def encode_labels(labels):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    return to_categorical(y_encoded), label_encoder