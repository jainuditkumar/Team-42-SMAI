#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cricket Bowler Type Recommendation with Transformer Model
This script implements a Transformer model to recommend cricket bowler types based on
match conditions including weather, pitch, venue, and match phase.
"""

import os
import argparse
import pandas as pd # type: ignore
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import Dataset, DataLoader # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from datetime import datetime
import json
import warnings
import pickle

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CricketDataset(Dataset):
    """Custom Dataset for Cricket match condition data"""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer"""
    
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class BowlerTypeTransformer(nn.Module):
    """Transformer model for recommending bowler types"""
    
    def __init__(self, input_dim, output_dim, d_model=64, nhead=4, num_layers=2, 
                 dim_feedforward=256, dropout=0.1):
        super(BowlerTypeTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout))
        self.transformer_encoder = nn.ModuleList(encoder_layers)
        
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = x.unsqueeze(0)
        x = self.pos_encoder(x)
        
        for layer in self.transformer_encoder:
            x = layer(x)
        
        x = x.squeeze(0)
        x = self.output_projection(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EarlyStopping:
    """Early stopping implementation to prevent overfitting"""
    
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class BowlerTypePerformanceAnalyzer:
    """Class to calculate bowler type performance metrics under various conditions"""
    
    def __init__(self):
        self.type_venue_economy = {}
        self.type_soil_economy = {}
        self.type_phase_economy = {}
        self.type_weather_economy = {}
        
    def analyze_historical_performance(self, df):
        print("Analyzing historical performance by bowler type...")
        
        if 'over_type' not in df.columns:
            df['over_type'] = pd.cut(df['overnumber'], 
                                 bins=[0, 6, 15, 20], 
                                 labels=['powerplay', 'middle', 'death'], 
                                 include_lowest=True)
        
        df['temp_category'] = pd.cut(df['temperature'], 
                                 bins=[0, 20, 30, 50], 
                                 labels=['cool', 'moderate', 'hot'], 
                                 include_lowest=True)
        
        df['humidity_category'] = pd.cut(df['humidity'], 
                                     bins=[0, 40, 70, 100], 
                                     labels=['low', 'medium', 'high'], 
                                     include_lowest=True)
        
        df['wind_category'] = pd.cut(df['wind_speed'], 
                                 bins=[0, 10, 20, 100], 
                                 labels=['light', 'moderate', 'strong'], 
                                 include_lowest=True)
        
        # Calculate economy as total_runs / overs (each row is an over)
        def calculate_economy(runs, overs, min_overs=1):
            if overs < min_overs:
                return np.nan
            economy = runs / overs
            return np.clip(economy, 0, 20)  # Cap economy to realistic range
        
        # Venue-specific economy
        type_venue_runs = df.groupby(['venue', 'bowler_type'])['total_runs'].sum().reset_index()
        type_venue_overs = df.groupby(['venue', 'bowler_type']).size().reset_index(name='overs')
        type_venue_stats = pd.merge(type_venue_runs, type_venue_overs, on=['venue', 'bowler_type'])
        type_venue_stats['economy'] = type_venue_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_venue_stats['economy'] = type_venue_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_venue_economy = type_venue_stats.set_index(['bowler_type', 'venue'])['economy'].to_dict()
        
        # Soil-specific economy
        type_soil_runs = df.groupby(['soil_type', 'bowler_type'])['total_runs'].sum().reset_index()
        type_soil_overs = df.groupby(['soil_type', 'bowler_type']).size().reset_index(name='overs')
        type_soil_stats = pd.merge(type_soil_runs, type_soil_overs, on=['soil_type', 'bowler_type'])
        type_soil_stats['economy'] = type_soil_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_soil_stats['economy'] = type_soil_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_soil_economy = type_soil_stats.set_index(['bowler_type', 'soil_type'])['economy'].to_dict()
        
        # Phase-specific economy
        type_phase_runs = df.groupby(['over_type', 'bowler_type'])['total_runs'].sum().reset_index()
        type_phase_overs = df.groupby(['over_type', 'bowler_type']).size().reset_index(name='overs')
        type_phase_stats = pd.merge(type_phase_runs, type_phase_overs, on=['over_type', 'bowler_type'])
        type_phase_stats['economy'] = type_phase_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_phase_stats['economy'] = type_phase_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_phase_economy = type_phase_stats.set_index(['bowler_type', 'over_type'])['economy'].to_dict()
        
        # Weather-specific economy (temperature)
        type_temp_runs = df.groupby(['temp_category', 'bowler_type'])['total_runs'].sum().reset_index()
        type_temp_overs = df.groupby(['temp_category', 'bowler_type']).size().reset_index(name='overs')
        type_temp_stats = pd.merge(type_temp_runs, type_temp_overs, on=['temp_category', 'bowler_type'])
        type_temp_stats['economy'] = type_temp_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_temp_stats['economy'] = type_temp_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_weather_economy['temp'] = type_temp_stats.set_index(['bowler_type', 'temp_category'])['economy'].to_dict()
        
        # Weather-specific economy (humidity)
        type_humidity_runs = df.groupby(['humidity_category', 'bowler_type'])['total_runs'].sum().reset_index()
        type_humidity_overs = df.groupby(['humidity_category', 'bowler_type']).size().reset_index(name='overs')
        type_humidity_stats = pd.merge(type_humidity_runs, type_humidity_overs, on=['humidity_category', 'bowler_type'])
        type_humidity_stats['economy'] = type_humidity_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_humidity_stats['economy'] = type_humidity_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_weather_economy['humidity'] = type_humidity_stats.set_index(['bowler_type', 'humidity_category'])['economy'].to_dict()
        
        # Weather-specific economy (wind)
        type_wind_runs = df.groupby(['wind_category', 'bowler_type'])['total_runs'].sum().reset_index()
        type_wind_overs = df.groupby(['wind_category', 'bowler_type']).size().reset_index(name='overs')
        type_wind_stats = pd.merge(type_wind_runs, type_wind_overs, on=['wind_category', 'bowler_type'])
        type_wind_stats['economy'] = type_wind_stats.apply(
            lambda row: calculate_economy(row['total_runs'], row['overs']), axis=1
        )
        type_wind_stats['economy'] = type_wind_stats['economy'].fillna(df['averagerunsperover'].mean())
        self.type_weather_economy['wind'] = type_wind_stats.set_index(['bowler_type', 'wind_category'])['economy'].to_dict()
        
        print("Performance analysis completed!")
        
        return {
            'type_venue_stats': type_venue_stats,
            'type_soil_stats': type_soil_stats,
            'type_phase_stats': type_phase_stats,
            'type_temp_stats': type_temp_stats,
            'type_humidity_stats': type_humidity_stats,
            'type_wind_stats': type_wind_stats
        }


class CricketModelManager:
    """Manager class to handle data processing, training, and evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.feature_cols = None
        self.label_encoder = None
        self.feature_scaler = None
        self.class_names = None
        self.feature_dim = None
        self.output_dim = None
        self.experiment_dir = None
        self.performance_analyzer = BowlerTypePerformanceAnalyzer()
        self.bowler_types = None
        
    def setup_experiment_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(self.config.output_dir, f"experiment_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        with open(os.path.join(self.experiment_dir, "config.json"), "w") as f:
            config_dict = vars(self.config)
            json.dump(config_dict, f, indent=4)
            
        return self.experiment_dir
    
    def preprocess_data(self, data_file):
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        
        self.bowler_types = df['bowler_type'].unique()
        print(f"Bowler types found: {self.bowler_types}")
        
        performance_data = self.performance_analyzer.analyze_historical_performance(df)
        self.plot_performance_analysis(performance_data)
        
        if 'over_type' not in df.columns:
            df['over_type'] = pd.cut(df['overnumber'], 
                                 bins=[0, 6, 15, 20], 
                                 labels=['powerplay', 'middle', 'death'], 
                                 include_lowest=True)
        
        df['venue_fast_economy'] = df.apply(
            lambda row: self.get_type_venue_economy('fast', row['venue']), axis=1
        )
        df['venue_spin_economy'] = df.apply(
            lambda row: self.get_type_venue_economy('spin', row['venue']), axis=1
        )
        df['venue_medium_economy'] = df.apply(
            lambda row: self.get_type_venue_economy('medium', row['venue']), axis=1
        )
        df['soil_fast_economy'] = df.apply(
            lambda row: self.get_type_soil_economy('fast', row['soil_type']), axis=1
        )
        df['soil_spin_economy'] = df.apply(
            lambda row: self.get_type_soil_economy('spin', row['soil_type']), axis=1
        )
        df['soil_medium_economy'] = df.apply(
            lambda row: self.get_type_soil_economy('medium', row['soil_type']), axis=1
        )
        df['phase_fast_economy'] = df.apply(
            lambda row: self.get_type_phase_economy('fast', row['over_type']), axis=1
        )
        df['phase_spin_economy'] = df.apply(
            lambda row: self.get_type_phase_economy('spin', row['over_type']), axis=1
        )
        df['phase_medium_economy'] = df.apply(
            lambda row: self.get_type_phase_economy('medium', row['over_type']), axis=1
        )
        
        df['temp_category'] = pd.cut(df['temperature'], 
                                 bins=[0, 20, 30, 50], 
                                 labels=['cool', 'moderate', 'hot'], 
                                 include_lowest=True)
        df['humidity_category'] = pd.cut(df['humidity'], 
                                     bins=[0, 40, 70, 100], 
                                     labels=['low', 'medium', 'high'], 
                                     include_lowest=True)
        df['wind_category'] = pd.cut(df['wind_speed'], 
                                 bins=[0, 10, 20, 100], 
                                 labels=['light', 'moderate', 'strong'], 
                                 include_lowest=True)
        
        df['temp_fast_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('fast', 'temp', row['temp_category']), axis=1
        )
        df['temp_spin_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('spin', 'temp', row['temp_category']), axis=1
        )
        df['temp_medium_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('medium', 'temp', row['temp_category']), axis=1
        )
        df['humidity_fast_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('fast', 'humidity', row['humidity_category']), axis=1
        )
        df['humidity_spin_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('spin', 'humidity', row['humidity_category']), axis=1
        )
        df['humidity_medium_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('medium', 'humidity', row['humidity_category']), axis=1
        )
        df['wind_fast_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('fast', 'wind', row['wind_category']), axis=1
        )
        df['wind_spin_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('spin', 'wind', row['wind_category']), axis=1
        )
        df['wind_medium_economy'] = df.apply(
            lambda row: self.get_type_weather_economy('medium', 'wind', row['wind_category']), axis=1
        )
        
        weather_features = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'cloud_cover']
        economy_features = [
            'venue_fast_economy', 'venue_spin_economy', 'venue_medium_economy',
            'soil_fast_economy', 'soil_spin_economy', 'soil_medium_economy',
            'phase_fast_economy', 'phase_spin_economy', 'phase_medium_economy',
            'temp_fast_economy', 'temp_spin_economy', 'temp_medium_economy',
            'humidity_fast_economy', 'humidity_spin_economy', 'humidity_medium_economy',
            'wind_fast_economy', 'wind_spin_economy', 'wind_medium_economy'
        ]
        
        print("Encoding categorical features...")
        soil_dummies = pd.get_dummies(df['soil_type'], prefix='soil')
        venue_dummies = pd.get_dummies(df['venue'], prefix='venue')
        over_dummies = pd.get_dummies(df['over_type'], prefix='phase')
        
        self.feature_cols = weather_features + economy_features + list(soil_dummies.columns) + list(venue_dummies.columns) + list(over_dummies.columns)
        
        features_df = pd.concat([
            df[weather_features + economy_features],
            soil_dummies,
            venue_dummies,
            over_dummies
        ], axis=1)
        
        self.label_encoder = LabelEncoder()
        df['target'] = self.label_encoder.fit_transform(df['bowler_type'])
        self.class_names = self.label_encoder.classes_
        self.output_dim = len(self.class_names)
        
        print(f"Target classes: {self.class_names}")
        
        self.feature_scaler = StandardScaler()
        X = self.feature_scaler.fit_transform(features_df)
        y = df['target'].values
        
        self.feature_dim = X.shape[1]
        print(f"Feature dimension: {self.feature_dim}")
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=self.config.seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=self.config.seed)
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        train_dataset = CricketDataset(X_train, y_train)
        val_dataset = CricketDataset(X_val, y_val)
        test_dataset = CricketDataset(X_test, y_test)
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_type_venue_economy(self, bowler_type, venue):
        type_mapping = {
            'fast': ['Right arm Fast', 'Left arm Fast'],
            'spin': ['Right arm Spin', 'Left arm Spin'],
            'medium': ['Right arm Medium', 'Left arm Medium']
        }
        if bowler_type in type_mapping:
            bowler_types = type_mapping[bowler_type]
            economies = [self.performance_analyzer.type_venue_economy.get((bt, venue), 0) for bt in bowler_types]
            valid_economies = [e for e in economies if e != 0]
            return np.mean(valid_economies) if valid_economies else 0
        return 0
    
    def get_type_soil_economy(self, bowler_type, soil_type):
        type_mapping = {
            'fast': ['Right arm Fast', 'Left arm Fast'],
            'spin': ['Right arm Spin', 'Left arm Spin'],
            'medium': ['Right arm Medium', 'Left arm Medium']
        }
        if bowler_type in type_mapping:
            bowler_types = type_mapping[bowler_type]
            economies = [self.performance_analyzer.type_soil_economy.get((bt, soil_type), 0) for bt in bowler_types]
            valid_economies = [e for e in economies if e != 0]
            return np.mean(valid_economies) if valid_economies else 0
        return 0
    
    def get_type_phase_economy(self, bowler_type, phase):
        type_mapping = {
            'fast': ['Right arm Fast', 'Left arm Fast'],
            'spin': ['Right arm Spin', 'Left arm Spin'],
            'medium': ['Right arm Medium', 'Left arm Medium']
        }
        if bowler_type in type_mapping:
            bowler_types = type_mapping[bowler_type]
            economies = [self.performance_analyzer.type_phase_economy.get((bt, phase), 0) for bt in bowler_types]
            valid_economies = [e for e in economies if e != 0]
            return np.mean(valid_economies) if valid_economies else 0
        return 0
    
    def get_type_weather_economy(self, bowler_type, weather_type, category):
        type_mapping = {
            'fast': ['Right arm Fast', 'Left arm Fast'],
            'spin': ['Right arm Spin', 'Left arm Spin'],
            'medium': ['Right arm Medium', 'Left arm Medium']
        }
        if bowler_type in type_mapping:
            bowler_types = type_mapping[bowler_type]
            economies = [self.performance_analyzer.type_weather_economy[weather_type].get((bt, category), 0) for bt in bowler_types]
            valid_economies = [e for e in economies if e != 0]
            return np.mean(valid_economies) if valid_economies else 0
        return 0
    
    def plot_performance_analysis(self, performance_data):
        performance_dir = os.path.join(self.experiment_dir, "performance_analysis")
        os.makedirs(performance_dir, exist_ok=True)
        
        # Convert bowler_type to string to ensure consistent hue typing
        for key in ['type_soil_stats', 'type_phase_stats', 'type_temp_stats', 
                    'type_humidity_stats', 'type_wind_stats']:
            performance_data[key]['bowler_type'] = performance_data[key]['bowler_type'].astype(str)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='soil_type', y='economy', hue='bowler_type', data=performance_data['type_soil_stats'])
        plt.title('Bowler Type Performance by Soil Type')
        plt.xlabel('Soil Type')
        plt.ylabel('Economy Rate')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(performance_dir, "soil_performance.png"))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='over_type', y='economy', hue='bowler_type', data=performance_data['type_phase_stats'])
        plt.title('Bowler Type Performance by Match Phase')
        plt.xlabel('Match Phase')
        plt.ylabel('Economy Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(performance_dir, "phase_performance.png"))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='temp_category', y='economy', hue='bowler_type', data=performance_data['type_temp_stats'])
        plt.title('Bowler Type Performance by Temperature')
        plt.xlabel('Temperature')
        plt.ylabel('Economy Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(performance_dir, "temperature_performance.png"))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='humidity_category', y='economy', hue='bowler_type', data=performance_data['type_humidity_stats'])
        plt.title('Bowler Type Performance by Humidity')
        plt.xlabel('Humidity')
        plt.ylabel('Economy Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(performance_dir, "humidity_performance.png"))
        plt.close()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='wind_category', y='economy', hue='bowler_type', data=performance_data['type_wind_stats'])
        plt.title('Bowler Type Performance by Wind Condition')
        plt.xlabel('Wind Strength')
        plt.ylabel('Economy Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(performance_dir, "wind_performance.png"))
        plt.close()
        
        for bowler_type in self.bowler_types:
            type_venues = performance_data['type_venue_stats'][performance_data['type_venue_stats']['bowler_type'] == bowler_type]
            top_venues = type_venues.sort_values('economy').head(10)
            
            plt.figure(figsize=(14, 8))
            sns.barplot(x='venue', y='economy', data=top_venues)
            plt.title(f'Top 10 Venues for {bowler_type}')
            plt.xlabel('Venue')
            plt.ylabel('Economy Rate')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(performance_dir, f"{bowler_type}_top_venues.png"))
            plt.close()
    
    def build_model(self):
        self.model = BowlerTypeTransformer(
            input_dim=self.feature_dim,
            output_dim=self.output_dim,
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            num_layers=self.config.num_layers,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout
        ).to(device)
        
        return self.model
    
    def train(self):
        print("Starting training...")
        
        self.build_model()
        
        # Compute class weights using square root of inverse frequency
        from collections import Counter
        import math
        train_labels = []
        for _, labels in self.train_dataloader:
            train_labels.extend(labels.numpy())
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = [math.sqrt(total_samples / class_counts[i]) if class_counts[i] > 0 else 1.0 for i in range(len(self.class_names))]
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        # Learning rate warmup scheduler
        warmup_epochs = 5
        base_lr = self.config.learning_rate
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (0.1 + 0.9 * min(epoch / warmup_epochs, 1.0)) if epoch < warmup_epochs else 1.0
        )
        
        # Plateau scheduler after warmup
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        checkpoint_path = os.path.join(self.experiment_dir, "best_model.pth")
        early_stopping = EarlyStopping(
            patience=self.config.patience,
            verbose=True,
            path=checkpoint_path
        )
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(self.config.epochs):
            self.model.train()
            train_loss = 0.0
            
            for features, labels in self.train_dataloader:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * features.size(0)
            
            train_loss = train_loss / len(self.train_dataloader.dataset)
            train_losses.append(train_loss)
            
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for features, labels in self.val_dataloader:
                    features, labels = features.to(device), labels.to(device)
                    
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * features.size(0)
                    
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(self.val_dataloader.dataset)
            val_losses.append(val_loss)
            
            val_accuracy = accuracy_score(val_true, val_preds)
            val_accuracies.append(val_accuracy)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.4f}")
            
            # Apply warmup scheduler for first warmup_epochs, then plateau scheduler
            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(val_loss)
            
            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        self.model.load_state_dict(torch.load(checkpoint_path))
        
        self.plot_training_curves(train_losses, val_losses, val_accuracies)
        
        return train_losses, val_losses, val_accuracies
    
    def evaluate(self, dataloader, split_name="Test"):
        self.model.eval()
        
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(device), labels.to(device)
                
                outputs = self.model(features)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_true, all_preds)
        report = classification_report(all_true, all_preds, target_names=self.class_names, output_dict=True)
        
        print(f"\n{split_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(all_true, all_preds, target_names=self.class_names))
        
        self.plot_confusion_matrix(all_true, all_preds, split_name)
        
        results = {
            "accuracy": float(accuracy),
            "classification_report": report,
            "predictions": all_preds if isinstance(all_preds, list) else all_preds.tolist(),
            "true_labels": all_true if isinstance(all_true, list) else all_true.tolist()
        }
        
        def default_converter(o):
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            elif isinstance(o, np.ndarray):
                return o.tolist()
            return str(o)
        
        results_path = os.path.join(self.experiment_dir, f"{split_name.lower()}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4, default=default_converter)
        
        return accuracy, report, all_preds, all_true
    
    def plot_training_curves(self, train_losses, val_losses, val_accuracies):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.experiment_dir, "training_curves.png"))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, split_name):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{split_name} Confusion Matrix')
        plt.savefig(os.path.join(self.experiment_dir, f"{split_name.lower()}_confusion_matrix.png"))
        plt.close()
    
    def save_model(self, path=None):
        if path is None:
            path = os.path.join(self.experiment_dir, "final_model.pth")
        
        model_info = {
            "state_dict": self.model.state_dict(),
            "feature_dim": self.feature_dim,
            "output_dim": self.output_dim,
            "d_model": self.config.d_model,
            "nhead": self.config.nhead,
            "num_layers": self.config.num_layers,
            "dim_feedforward": self.config.dim_feedforward,
            "dropout": self.config.dropout,
            "class_names": self.class_names.tolist() if hasattr(self.class_names, 'tolist') else self.class_names,
            "feature_cols": self.feature_cols,
        }
        
        torch.save(model_info, path)
        print(f"Model saved to {path}")
        
        scaler_path = os.path.join(self.experiment_dir, "feature_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        
        label_encoder_path = os.path.join(self.experiment_dir, "label_encoder.pkl")
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        economy_path = os.path.join(self.experiment_dir, "economy_data.pkl")
        with open(economy_path, 'wb') as f:
            pickle.dump({
                "type_venue_economy": self.performance_analyzer.type_venue_economy,
                "type_soil_economy": self.performance_analyzer.type_soil_economy,
                "type_phase_economy": self.performance_analyzer.type_phase_economy,
                "type_weather_economy": self.performance_analyzer.type_weather_economy
            }, f)
        
        return path
    
    def load_model(self, path):
        model_info = torch.load(path, map_location=device)
        
        self.feature_dim = model_info["feature_dim"]
        self.output_dim = model_info["output_dim"]
        self.class_names = model_info["class_names"]
        self.feature_cols = model_info["feature_cols"]
        
        self.model = BowlerTypeTransformer(
            input_dim=self.feature_dim,
            output_dim=self.output_dim,
            d_model=model_info["d_model"],
            nhead=model_info["nhead"],
            num_layers=model_info["num_layers"],
            dim_feedforward=model_info["dim_feedforward"],
            dropout=model_info["dropout"]
        ).to(device)
        
        self.model.load_state_dict(model_info["state_dict"])
        print(f"Model loaded from {path}")
        
        scaler_path = os.path.join(os.path.dirname(path), "feature_scaler.pkl")
        with open(scaler_path, 'rb') as f:
            self.feature_scaler = pickle.load(f)
        
        label_encoder_path = os.path.join(os.path.dirname(path), "label_encoder.pkl")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        economy_path = os.path.join(os.path.dirname(path), "economy_data.pkl")
        with open(economy_path, 'rb') as f:
            economy_data = pickle.load(f)
            self.performance_analyzer.type_venue_economy = economy_data["type_venue_economy"]
            self.performance_analyzer.type_soil_economy = economy_data["type_soil_economy"]
            self.performance_analyzer.type_phase_economy = economy_data["type_phase_economy"]
            self.performance_analyzer.type_weather_economy = economy_data["type_weather_economy"]
        
        return self.model
    
    def predict(self, features):
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)  # Compute softmax probabilities
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            probabilities = probabilities.cpu().numpy()
        
        pred_classes = self.label_encoder.inverse_transform(preds)
        return pred_classes, probabilities


def train_model(args):
    print("Training model...")
    
    manager = CricketModelManager(args)
    manager.setup_experiment_dir()
    manager.preprocess_data(args.data_file)
    train_losses, val_losses, val_accuracies = manager.train()
    test_accuracy, test_report, test_preds, test_true = manager.evaluate(manager.test_dataloader, "Test")
    model_path = manager.save_model()
    
    print(f"Training completed. Model saved to {model_path}")
    print(f"Experiment directory: {manager.experiment_dir}")
    
    return test_accuracy


def test_model(args):
    print("Testing model...")
    
    manager = CricketModelManager(args)
    manager.load_model(args.model_path)
    manager.preprocess_data(args.data_file)
    test_accuracy, test_report, test_preds, test_true = manager.evaluate(manager.test_dataloader, "Test")
    
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return test_accuracy


def predict_with_model(args):
    print("Making predictions...")
    
    manager = CricketModelManager(args)
    manager.load_model(args.model_path)
    
    if args.predict_from_input:
        print("\nPredicting using user-provided values:")
        
        user_inputs = {
            'temperature': args.temperature,
            'humidity': args.humidity,
            'precipitation': args.precipitation,
            'wind_speed': args.wind_speed,
            'cloud_cover': args.cloud_cover,
            'soil_type': args.soil_type,
            'venue': args.venue,
            'overnumber': args.overnumber
        }
        
        print("\nInput features:")
        for key, value in user_inputs.items():
            print(f"{key}: {value}")
        
        df = pd.DataFrame([user_inputs])
        
        df['over_type'] = pd.cut(df['overnumber'], 
                               bins=[0, 6, 15, 20], 
                               labels=['powerplay', 'middle', 'death'], 
                               include_lowest=True)
        
        ref_df = pd.read_csv(args.data_file)
        
        if not (manager.performance_analyzer.type_venue_economy and 
                manager.performance_analyzer.type_soil_economy and 
                manager.performance_analyzer.type_phase_economy and 
                manager.performance_analyzer.type_weather_economy):
            print("\nCalculating economy references from data file...")
            manager.performance_analyzer.analyze_historical_performance(ref_df)
    else:
        print(f"Loading prediction data from {args.data_file}...")
        df = pd.read_csv(args.data_file)
        
        if 'over_type' not in df.columns:
            df['over_type'] = pd.cut(df['overnumber'], 
                                 bins=[0, 6, 15, 20], 
                                 labels=['powerplay', 'middle', 'death'], 
                                 include_lowest=True)
    
    # ... (Economy feature calculations remain unchanged)
    
    df['venue_fast_economy'] = df.apply(
        lambda row: manager.get_type_venue_economy('fast', row['venue']), axis=1
    )
    df['venue_spin_economy'] = df.apply(
        lambda row: manager.get_type_venue_economy('spin', row['venue']), axis=1
    )
    df['venue_medium_economy'] = df.apply(
        lambda row: manager.get_type_venue_economy('medium', row['venue']), axis=1
    )
    df['soil_fast_economy'] = df.apply(
        lambda row: manager.get_type_soil_economy('fast', row['soil_type']), axis=1
    )
    df['soil_spin_economy'] = df.apply(
        lambda row: manager.get_type_soil_economy('spin', row['soil_type']), axis=1
    )
    df['soil_medium_economy'] = df.apply(
        lambda row: manager.get_type_soil_economy('medium', row['soil_type']), axis=1
    )
    df['phase_fast_economy'] = df.apply(
        lambda row: manager.get_type_phase_economy('fast', row['over_type']), axis=1
    )
    df['phase_spin_economy'] = df.apply(
        lambda row: manager.get_type_phase_economy('spin', row['over_type']), axis=1
    )
    df['phase_medium_economy'] = df.apply(
        lambda row: manager.get_type_phase_economy('medium', row['over_type']), axis=1
    )
    
    df['temp_category'] = pd.cut(df['temperature'], 
                             bins=[0, 20, 30, 50], 
                             labels=['cool', 'moderate', 'hot'], 
                             include_lowest=True)
    df['humidity_category'] = pd.cut(df['humidity'], 
                                 bins=[0, 40, 70, 100], 
                                 labels=['low', 'medium', 'high'], 
                                 include_lowest=True)
    df['wind_category'] = pd.cut(df['wind_speed'], 
                             bins=[0, 10, 20, 100], 
                             labels=['light', 'moderate', 'strong'], 
                             include_lowest=True)
    
    df['temp_fast_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('fast', 'temp', row['temp_category']), axis=1
    )
    df['temp_spin_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('spin', 'temp', row['temp_category']), axis=1
    )
    df['temp_medium_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('medium', 'temp', row['temp_category']), axis=1
    )
    df['humidity_fast_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('fast', 'humidity', row['humidity_category']), axis=1
    )
    df['humidity_spin_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('spin', 'humidity', row['humidity_category']), axis=1
    )
    df['humidity_medium_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('medium', 'humidity', row['humidity_category']), axis=1
    )
    df['wind_fast_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('fast', 'wind', row['wind_category']), axis=1
    )
    df['wind_spin_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('spin', 'wind', row['wind_category']), axis=1
    )
    df['wind_medium_economy'] = df.apply(
        lambda row: manager.get_type_weather_economy('medium', 'wind', row['wind_category']), axis=1
    )
    
    features_df = pd.DataFrame(index=df.index, columns=manager.feature_cols)
    features_df = features_df.fillna(0)
    
    weather_features = ['temperature', 'humidity', 'precipitation', 'wind_speed', 'cloud_cover']
    economy_features = [
        'venue_fast_economy', 'venue_spin_economy', 'venue_medium_economy',
        'soil_fast_economy', 'soil_spin_economy', 'soil_medium_economy',
        'phase_fast_economy', 'phase_spin_economy', 'phase_medium_economy',
        'temp_fast_economy', 'temp_spin_economy', 'temp_medium_economy',
        'humidity_fast_economy', 'humidity_spin_economy', 'humidity_medium_economy',
        'wind_fast_economy', 'wind_spin_economy', 'wind_medium_economy'
    ]
    
    for col in weather_features + economy_features:
        if col in df.columns:
            features_df[col] = df[col]
    
    for idx, row in df.iterrows():
        soil_col = f"soil_{row['soil_type']}"
        if soil_col in features_df.columns:
            features_df.at[idx, soil_col] = 1
        
        venue_col = f"venue_{row['venue']}"
        if venue_col in features_df.columns:
            features_df.at[idx, venue_col] = 1
        
        phase_col = f"phase_{row['over_type']}"
        if phase_col in features_df.columns:
            features_df.at[idx, phase_col] = 1
    
    if args.predict_from_input:
        print("\nProcessed features for prediction:")
        for col in weather_features + economy_features:
            print(f"{col}: {features_df[col].values[0]}")
        for col in features_df.columns:
            if col not in weather_features + economy_features and features_df[col].values[0] == 1:
                print(f"{col}: {features_df[col].values[0]}")
    
    X = manager.feature_scaler.transform(features_df)
    predictions, probabilities = manager.predict(X)
    
    df['recommended_bowler_type'] = predictions
    
    if args.predict_from_input:
        print("\nPrediction Result:")
        print(f"Recommended Bowler Type: {predictions[0]}")
        if args.percentage:
            print("\nLikelihood of Each Bowler Type:")
            for class_name, prob in zip(manager.class_names, probabilities[0]):
                print(f"{class_name}: {prob*100:.2f}%")
    else:
        output_file = args.output_file
        df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        prediction_counts = df['recommended_bowler_type'].value_counts()
        print("\nPrediction Summary:")
        for bowler_type, count in prediction_counts.items():
            print(f"{bowler_type}: {count}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Cricket Bowler Type Recommendation with Transformer Model")
    subparsers = parser.add_subparsers(dest="mode", help="Mode to run")
    
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--data_file", type=str, default="final.csv", help="Path to the CSV data file")
    common_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    common_parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    
    train_parser = subparsers.add_parser("train", parents=[common_parser], help="Train a new model")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    train_parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")
    train_parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    train_parser.add_argument("--d_model", type=int, default=128, help="Dimension of the transformer model")
    train_parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    train_parser.add_argument("--dim_feedforward", type=int, default=256, help="Dimension of feedforward network")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    test_parser = subparsers.add_parser("test", parents=[common_parser], help="Test an existing model")
    test_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    test_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    
    predict_parser = subparsers.add_parser("predict", parents=[common_parser], help="Make predictions with an existing model")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model file")
    predict_parser.add_argument("--output_file", type=str, default="predictions.csv", help="Path to save predictions")
    predict_parser.add_argument("--predict_from_input", action="store_true", help="Predict from command line input instead of file")
    predict_parser.add_argument("--percentage", action="store_true", help="Show percentage likelihood of each bowler type")
    predict_parser.add_argument("--temperature", type=float, default=30.0, help="Temperature in Celsius")
    predict_parser.add_argument("--humidity", type=float, default=60.0, help="Humidity percentage")
    predict_parser.add_argument("--precipitation", type=float, default=0.0, help="Precipitation in mm")
    predict_parser.add_argument("--wind_speed", type=float, default=10.0, help="Wind speed in km/h")
    predict_parser.add_argument("--cloud_cover", type=float, default=20.0, help="Cloud cover percentage")
    predict_parser.add_argument("--soil_type", type=str, default="black", help="Soil type of the pitch")
    predict_parser.add_argument("--venue", type=str, default="Eden Gardens", help="Cricket venue name")
    predict_parser.add_argument("--overnumber", type=int, default=10, help="Over number (1-20)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)
    elif args.mode == "predict":
        predict_with_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()