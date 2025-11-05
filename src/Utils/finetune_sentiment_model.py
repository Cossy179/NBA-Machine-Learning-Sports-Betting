"""
Fine-tuning utility for BERT-based sentiment analysis on NBA-specific data.

This script allows you to fine-tune a pre-trained BERT model on a corpus of
NBA news articles, injury reports, and social media posts to improve
sentiment classification accuracy for sports betting predictions.

Usage:
    python src/Utils/finetune_sentiment_model.py --data_path data/nba_sentiment_corpus.csv
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple
import json
import os

try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, DataCollatorWithPadding
    )
    from datasets import Dataset
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers library not available. Install with: pip install transformers datasets")


class NBASentimentDataset:
    """Dataset handler for NBA sentiment fine-tuning"""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset from CSV file.
        
        Expected CSV columns:
        - text: The text content (article, post, etc.)
        - label: Sentiment label (0=negative, 1=neutral, 2=positive) or (positive/negative/neutral)
        - source: Source of the text (ESPN, Reddit, etc.)
        - date: Publication date
        """
        self.data_path = data_path
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load data from CSV file"""
        if not os.path.exists(self.data_path):
            print(f"⚠️  Data file not found: {self.data_path}")
            print("Creating sample dataset structure...")
            self._create_sample_dataset()
            return
        
        self.data = pd.read_csv(self.data_path)
        
        # Validate required columns
        required_cols = ['text', 'label']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Normalize labels
        self._normalize_labels()
        
        print(f"✅ Loaded {len(self.data)} samples")
        print(f"   Label distribution: {self.data['label'].value_counts().to_dict()}")
    
    def _normalize_labels(self):
        """Normalize labels to 0 (negative), 1 (neutral), 2 (positive)"""
        if self.data['label'].dtype == 'object':
            label_map = {
                'negative': 0, 'neg': 0, '0': 0,
                'neutral': 1, 'neu': 1, '1': 1,
                'positive': 2, 'pos': 2, '2': 2
            }
            self.data['label'] = self.data['label'].str.lower().map(label_map)
        
        # Ensure labels are integers
        self.data['label'] = self.data['label'].astype(int)
    
    def _create_sample_dataset(self):
        """Create a sample dataset structure for reference"""
        sample_data = {
            'text': [
                "Lakers win big against Celtics in overtime thriller",
                "Warriors star suffers season-ending injury",
                "Nuggets trade for All-Star point guard",
                "Knicks struggle in fourth quarter, lose by 15",
                "Bucks dominate defensively, hold opponent under 90 points"
            ],
            'label': [2, 0, 1, 0, 2],  # 0=negative, 1=neutral, 2=positive
            'source': ['ESPN', 'ESPN', 'Reddit', 'The Athletic', 'ESPN'],
            'date': [datetime.now()] * 5
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_path = self.data_path.replace('.csv', '_sample.csv')
        sample_df.to_csv(sample_path, index=False)
        print(f"📝 Created sample dataset: {sample_path}")
        print("   Update this file with your NBA sentiment data and rename to the original path.")
    
    def get_train_test_split(self, test_size: float = 0.2):
        """Split data into train and test sets"""
        from sklearn.model_selection import train_test_split
        
        train_data, test_data = train_test_split(
            self.data,
            test_size=test_size,
            stratify=self.data['label'],
            random_state=42
        )
        
        return train_data, test_data


def prepare_dataset_for_training(texts: List[str], labels: List[int], 
                                  tokenizer, max_length: int = 512) -> Dataset:
    """Prepare dataset for HuggingFace Trainer"""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library required for fine-tuning")
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    })
    
    return dataset


def fine_tune_model(base_model_name: str, train_dataset: Dataset, 
                   eval_dataset: Dataset, output_dir: str,
                   num_epochs: int = 3, batch_size: int = 16):
    """
    Fine-tune BERT model on NBA sentiment data.
    
    Args:
        base_model_name: HuggingFace model name (e.g., "cardiffnlp/twitter-roberta-base-sentiment-latest")
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers library required for fine-tuning")
    
    print(f"🚀 Starting fine-tuning of {base_model_name}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Evaluation samples: {len(eval_dataset)}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=3  # negative, neutral, positive
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("📚 Training model...")
    trainer.train()
    
    # Evaluate
    print("📊 Evaluating model...")
    eval_results = trainer.evaluate()
    print(f"   Evaluation accuracy: {eval_results.get('eval_accuracy', 0):.4f}")
    
    # Save model
    print(f"💾 Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Fine-tuning complete!")
    
    return model, tokenizer


def collect_nba_sentiment_data(output_path: str, num_samples: int = 1000):
    """
    Helper function to collect NBA sentiment data from various sources.
    This is a placeholder - in production, you would scrape/collect real data.
    
    Args:
        output_path: Path to save collected data CSV
        num_samples: Target number of samples to collect
    """
    print("📥 This function would collect NBA sentiment data from:")
    print("   - ESPN articles")
    print("   - The Athletic articles")
    print("   - Reddit r/NBA posts")
    print("   - Team press releases")
    print("   - Injury reports")
    print("\n   In production, implement actual scraping/API calls here.")
    print("   For now, manually label a corpus of NBA-related texts.")
    
    # Create sample structure
    sample_data = {
        'text': [],
        'label': [],
        'source': [],
        'date': []
    }
    
    # This would be populated with actual data collection logic
    # For now, create empty structure
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"\n📝 Created data collection template: {output_path}")
    print("   Manually populate this file with labeled NBA sentiment data.")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT model for NBA sentiment analysis')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV file with NBA sentiment data')
    parser.add_argument('--base_model', type=str, 
                       default='cardiffnlp/twitter-roberta-base-sentiment-latest',
                       help='Base HuggingFace model name')
    parser.add_argument('--output_dir', type=str, default='Models/SentimentModels/nba_bert_sentiment',
                       help='Directory to save fine-tuned model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--collect_data', action='store_true',
                       help='Collect sample data instead of training')
    
    args = parser.parse_args()
    
    if args.collect_data:
        collect_nba_sentiment_data(args.data_path)
        return
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers library not available.")
        print("   Install with: pip install transformers datasets torch")
        return
    
    # Load dataset
    dataset = NBASentimentDataset(args.data_path)
    
    if dataset.data is None or len(dataset.data) == 0:
        print("❌ No data available for training.")
        return
    
    # Split data
    train_data, test_data = dataset.get_train_test_split(test_size=0.2)
    
    # Prepare datasets
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    train_dataset = prepare_dataset_for_training(
        train_data['text'].tolist(),
        train_data['label'].tolist(),
        tokenizer
    )
    
    eval_dataset = prepare_dataset_for_training(
        test_data['text'].tolist(),
        test_data['label'].tolist(),
        tokenizer
    )
    
    # Fine-tune model
    fine_tune_model(
        base_model_name=args.base_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\n✅ Fine-tuned model saved to: {args.output_dir}")
    print("   Use this model path in SentimentAnalysis.py to load your custom model")


if __name__ == "__main__":
    main()

