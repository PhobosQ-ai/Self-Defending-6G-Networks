import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import kagglehub
from kagglehub import KaggleDatasetAdapter
import logging

logger = logging.getLogger(__name__)

def get_dataloader(batch_size: int = 64) -> tuple[DataLoader, int, int]:
    """
    Loads and preprocesses data from CIC-IoT-2023 dataset to create a DataLoader.
    This function loads the dataset, applies filtering, encoding, and normalization as per industry practices.
    
    Returns:
        dataloader: PyTorch DataLoader for the dataset
        num_features: Number of features in the dataset
        num_classes: Number of classes (attack types)
    """
    try:
        file_path = ""
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "akashdogra/cic-iot-2023",
            file_path,
        )
        
        if isinstance(df, dict):
            df = list(df.values())[0] if df else pd.DataFrame()
        
        if df.empty:
            raise ValueError("Loaded dataset is empty. Please check the file_path or dataset structure.")
        
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        df = keep_max_subclass(df)
        logger.info(f"After filtering: {df.shape}")
        
        features = df.select_dtypes(include=['number'])
        if 'label' in df.columns:
            labels = df['label']
        else:
            labels = pd.Series([0] * len(df))
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        features_scaled = scaler.fit_transform(features)
        
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        
        features_tensor = torch.FloatTensor(features_scaled)
        labels_tensor = torch.LongTensor(labels_encoded)
        
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        num_features = features_tensor.shape[1]
        num_classes = len(label_encoder.classes_)
        
        logger.info(f"Data loaded: {len(dataset)} samples, {num_features} features, {num_classes} classes.")
        
        return dataloader, num_features, num_classes
    
    except Exception as e:
        logger.error(f"Error in data loading: {e}")
        raise

def keep_max_subclass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the dataset to keep only the most frequent subclass for each main attack class,
    and maps them to main class labels for reduced complexity.
    Based on the notebook's implementation.
    """
    subclass_mapping = {
        'DDoS': ['DDoS-ICMP_Flood', 'DDoS-UDP_Flood', 'DDoS-TCP_Flood', 'DDoS-PSHACK_Flood',
                 'DDoS-SYN_Flood', 'DDoS-RSTFINFlood', 'DDoS-SynonymousIP_Flood',
                 'DDoS-ICMP_Fragmentation', 'DDoS-UDP_Fragmentation', 'DDoS-ACK_Fragmentation',
                 'DDoS-HTTP_Flood', 'DDoS-SlowLoris'],
        'DoS': ['DoS-UDP_Flood', 'DoS-TCP_Flood', 'DoS-SYN_Flood', 'DoS-HTTP_Flood'],
        'Recon': ['Recon-HostDiscovery', 'Recon-OSScan', 'Recon-PortScan',
                 'Recon-PingSweep', 'VulnerabilityScan'],
        'Spoofing': ['MITM-ArpSpoofing', 'DNS_Spoofing'],
        'BruteForce': ['DictionaryBruteForce'],
        'Web-based': ['BrowserHijacking', 'XSS', 'Uploading_Attack', 'SqlInjection',
                     'CommandInjection', 'Backdoor_Malware'],
        'Mirai': ['Mirai-greeth_flood', 'Mirai-udpplain', 'Mirai-greip_flood'],
        'BENIGN': ['BenignTraffic']
    }

    subclass_to_main = {}
    for main_class, subclasses in subclass_mapping.items():
        for subclass in subclasses:
            subclass_to_main[subclass] = main_class

    main_class_max_subclass = {}
    for main_class, subclasses in subclass_mapping.items():
        if main_class == 'BENIGN':
            continue
        subclass_counts = df[df['label'].isin(subclasses)]['label'].value_counts()
        if not subclass_counts.empty:
            max_subclass = subclass_counts.idxmax()
            main_class_max_subclass[max_subclass] = main_class

    mask = df['label'].isin(main_class_max_subclass.keys()) | (df['label'] == 'BenignTraffic')
    filtered_df = df[mask].copy()
    for subclass, main_class in main_class_max_subclass.items():
        filtered_df.loc[filtered_df['label'] == subclass, 'label'] = main_class

    return filtered_df
