import os
import io
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import load, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from nnets import BackboneSFM
from utils import custom_print

RANDOM_SEED = 678
torch.manual_seed(RANDOM_SEED)

class EmotionDataset(Dataset, BackboneSFM):
    """
    Args:
        params: config object (must contain dataroot, sample_rate, csv_path, etc.)
        df (pd.DataFrame, optional): DataFrame with cols 'audio_path'/'audio' and 'emotion'. If not defined, load from params.csv_path.
        feature_extractor: feature extraction object (e.g., Wav2Vec2FeatureExtractor).
    """

    def __init__(self, params, df=None):
        super().__init__(params)
        
        # Load Dataframe
        if self.params.csv_path is not None and df is None:
            print("Create dataset from CSV.")
            self.df = pd.read_csv(self.params.csv_path)
        else:
            print("\nCreate Dataset from Dataframe.")
            self.df = df.reset_index(drop=True)
        self.emotion_labels = self._prepare_emotion_labels()
        # self.augment_wav = AugmentWAV(self.params.noise_path, self.params.rir_path) # TODO?

    def _prepare_emotion_labels(self):
        """Convert textual labels to numerical"""

        unique_emotions = self.df['emotion'].unique()
        unique_labels = self.df['label'].unique()
        res = {}
        for e, l in zip(unique_emotions, unique_labels):
            print(f"{e}:{l}")
            res[e] = l
        return res

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load audio data
        if 'audio' not in row.index:
            audio_path = os.path.join(self.params.dataroot, row['audio_path'])
            if isinstance(audio_path, str):
                waveform, sample_rate = load(audio_path)
            else:
                raise KeyError("Neither 'audio' column nor 'audio_path'/'file' found in DataFrame.")
        else:
            buffer = io.BytesIO(row["audio"]["bytes"])
            waveform, sample_rate = load(buffer)
        emotion = row["emotion"]

        # Resampling
        if sample_rate != self.params.sample_rate:
            resampler = transforms.Resample(orig_freq=sample_rate, new_freq=self.params.sample_rate)
            waveform = resampler(waveform)

        # Normalization
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

        # Convert to float32 tensor
        if str(waveform.dtype) != 'torch.float32': waveform = torch.from_numpy(waveform)

        # Conver to mono
        if waveform.shape[0] != 1: waveform = waveform.mean(0)[None ,:]

        # Fill with copy (min duration 5 seconds)
        min_dur = self.params.sample_rate * 5
        if waveform.shape[1] < min_dur:
            while waveform.shape[1] < min_dur:
                waveform = torch.cat([waveform, waveform], 1)

        # Feature extraction
        inputs = self.feature_extractor(
            waveform.squeeze(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.params.sample_rate * 4,  # restrict duration
        ).input_values
        inputs = inputs.squeeze(0)
        label = self.emotion_labels[emotion]

        return inputs, label, emotion, idx


def get_stratified_data_splits(df_full, ratio_train=0.9, ratio_test=0.1, seed=RANDOM_SEED):
    """
    Stratified split Dataframe to train and test subset by label.
    Args:
        df_full (pd.DataFrame)
        ratio_train (float)
        ratio_test (float)
    Returns:
        tuple: (df_train, df_test)
    """

    assert abs(ratio_train + ratio_test - 1.0) < 1e-6, "Split ratio sum must be equal 1.0!"
    df_train, df_test = train_test_split(df_full, train_size=ratio_train, stratify=df_full['label'], random_state=seed)
    return df_train, df_test


def get_dataloader(dataset, batch_size, n_workers, shuffle=True):
    """
    Create DataLoader with custom collate function.
    Args:
        dataset (Dataset)
        batch_size (int)
        n_workers (int)
        shuffle (bool)
    Returns:
        DataLoader
    """

    def _collate_fn(batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.stack(inputs), torch.tensor(labels)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_fn,
        num_workers=n_workers)
    return data_loader


def get_dataframe(args):

    """
    Load and prepare DataFrame.
    Args:
        args: config object (must contain dataroot, dataset_name, num_classes)
    Returns:
        pd.DataFrame: prepared DataFrame with 'audio_path', 'emotion', 'label'.
    """

    custom_print("\nLoad data...")
    ds = load_dataset(args.dataroot)
    df_full = pd.concat([split.to_pandas() for split in ds.values()], ignore_index=True)
    
    if args.dataset_name == "iemocap":
        emotion_col = "major_emotion"
        emotion_mapping = {
            2: ["happy", "sad"],
            3: ["neutral", "happy", "sad"],
            4: ["excited", "neutral", "angry", "sad"],
            5: ["frustrated", "excited", "neutral", "angry", "sad"],
            6: ["frustrated", "excited", "neutral", "angry", "sad", "happy"],
            8: ["frustrated", "excited", "neutral", "angry", "sad", "happy", "surprise", "fear"]
        }
    elif args.dataset_name == "emotiontalk":
        args.dataroot = os.path.join(args.dataroot, "Audio/wav")
        emotion_col = "emotion_result"
        emotion_mapping = {
            4: ["neutral", "angry", "happy", "surprised"],
            5: ["neutral", "angry", "happy", "surprised", "sad"],
        }
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")
    
    # Statistics by emotions
    emotion_counts = df_full[emotion_col].value_counts()
    custom_print(f"\n{args.dataset_name} data distribution:\n{emotion_counts}\n")

    # Filtration by number of classes
    selected_emotions = emotion_mapping.get(args.num_classes, list(emotion_counts.index))
    df_filtered = df_full[df_full[emotion_col].isin(selected_emotions)]
    
    # Prepare IEMOCAP
    if args.dataset_name == "iemocap":
        df_filtered = df_filtered.rename(columns={'file': 'audio_path', emotion_col: 'emotion'})
        keep_cols = ['audio_path', 'audio', 'emotion']
        df_filtered = df_filtered[[c for c in keep_cols if c in df_filtered.columns]]
    # Prepare EMOTIONTALK
    elif args.dataset_name == "emotiontalk":
        df_filtered = df_filtered.rename(columns={'file_path': 'audio_path', emotion_col: 'emotion'})
        df_filtered = df_filtered[['audio_path', 'emotion']]
    
    # Check the number of unique emotions
    unique_emotions = set(df_filtered['emotion'])
    assert len(unique_emotions) == args.num_classes, \
        f"Expected {args.num_classes} classes, got {len(unique_emotions)}: {unique_emotions}"

    # Encode labels
    le = LabelEncoder()
    df_filtered['label'] = le.fit_transform(df_filtered['emotion'])

    for i, emo in enumerate(le.classes_):
        custom_print(f"{i}: {emo}, {emotion_counts.get(emo, 0)} files")

    return df_filtered
