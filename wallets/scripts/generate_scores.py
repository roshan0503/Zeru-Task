#!/usr/bin/env python3
"""
Aave V2 Credit Score Generation Script
=====================================

This script generates credit scores (0-1000) for wallet addresses based on 
their historical transaction behavior in the Aave V2 protocol.

Usage:
    python generate_scores.py

Requirements:
    - user-wallet-transactions.json file in the same directory
    - See requirements.txt for Python dependencies
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AaveCreditScorer:
    """
    Credit scoring system for Aave V2 wallets based on transaction behavior.
    
    The scoring system uses a weighted combination of features that indicate
    responsible vs risky behavior in DeFi lending protocols.
    """
    
    def __init__(self):
        self.feature_weights = {
            # Positive indicators (responsible behavior)
            'repay_to_borrow_ratio': 200,
            'deposit_consistency': 150,
            'repayment_frequency': 100,
            'portfolio_diversification': 50,
            'activity_duration_score': 75,
            'transaction_regularity': 50,
            
            # Negative indicators (risky behavior)
            'liquidation_penalty': -400,
            'borrowing_intensity': -100,
            'high_risk_asset_exposure': -50,
            'bot_like_behavior': -150,
        }
        
        self.base_score = 500  # Starting score
        self.min_score = 0
        self.max_score = 1000
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load and preprocess transaction data."""
        print("Loading transaction data...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Extract action data fields
        action_data_df = pd.json_normalize(df['actionData'])
        df = pd.concat([df, action_data_df], axis=1)
        
        # Convert amount to float and handle missing values
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
        df['assetPriceUSD'] = pd.to_numeric(df['assetPriceUSD'], errors='coerce').fillna(0)
        
        # Calculate USD amount
        df['amountUSD'] = df['amount'] * df['assetPriceUSD']
        
        print(f"Loaded {len(df)} transactions from {df['userWallet'].nunique()} unique wallets")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for each wallet based on transaction history."""
        print("Engineering features...")
        
        features = []
        
        for wallet in df['userWallet'].unique():
            wallet_data = df[df['userWallet'] == wallet].copy()
            wallet_data = wallet_data.sort_values('timestamp')
            
            feature_dict = {'userWallet': wallet}
            
            # Basic transaction metrics
            feature_dict.update(self._calculate_transaction_metrics(wallet_data))
            
            # Financial metrics
            feature_dict.update(self._calculate_financial_metrics(wallet_data))
            
            # Time-based metrics
            feature_dict.update(self._calculate_time_metrics(wallet_data))
            
            # Risk indicators
            feature_dict.update(self._calculate_risk_indicators(wallet_data))
            
            # Advanced behavioral features
            feature_dict.update(self._calculate_behavioral_features(wallet_data))
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _calculate_transaction_metrics(self, wallet_data: pd.DataFrame) -> Dict:
        """Calculate basic transaction count and type metrics."""
        metrics = {}
        
        # Total transactions
        metrics['total_transactions'] = len(wallet_data)
        
        # Action type counts
        action_counts = wallet_data['action'].value_counts()
        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
            metrics[f'{action}_count'] = action_counts.get(action, 0)
        
        # Action ratios
        borrow_count = metrics['borrow_count']
        repay_count = metrics['repay_count']
        
        metrics['repay_to_borrow_ratio'] = repay_count / max(borrow_count, 1)
        metrics['liquidation_to_total_ratio'] = metrics['liquidationcall_count'] / metrics['total_transactions']
        
        return metrics
    
    def _calculate_financial_metrics(self, wallet_data: pd.DataFrame) -> Dict:
        """Calculate financial behavior metrics."""
        metrics = {}
        
        # Total amounts by action
        for action in ['deposit', 'borrow', 'repay', 'redeemunderlying']:
            action_data = wallet_data[wallet_data['action'] == action]
            metrics[f'total_{action}_usd'] = action_data['amountUSD'].sum()
        
        # Average transaction sizes
        metrics['avg_transaction_usd'] = wallet_data['amountUSD'].mean()
        metrics['max_transaction_usd'] = wallet_data['amountUSD'].max()
        
        # Portfolio metrics
        metrics['unique_assets'] = wallet_data['assetSymbol'].nunique()
        
        # Borrowing metrics
        borrow_data = wallet_data[wallet_data['action'] == 'borrow']
        if len(borrow_data) > 0:
            metrics['avg_borrow_rate'] = pd.to_numeric(borrow_data['borrowRate'], errors='coerce').mean()
            metrics['max_borrow_rate'] = pd.to_numeric(borrow_data['borrowRate'], errors='coerce').max()
        else:
            metrics['avg_borrow_rate'] = 0
            metrics['max_borrow_rate'] = 0
        
        return metrics
    
    def _calculate_time_metrics(self, wallet_data: pd.DataFrame) -> Dict:
        """Calculate time-based behavior metrics."""
        metrics = {}
        
        if len(wallet_data) < 2:
            metrics['activity_duration_days'] = 0
            metrics['avg_time_between_tx_hours'] = 0
            metrics['transaction_frequency_per_day'] = 0
        else:
            # Activity duration
            duration = (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).total_seconds()
            metrics['activity_duration_days'] = duration / (24 * 3600)
            
            # Average time between transactions
            time_diffs = wallet_data['timestamp'].diff().dt.total_seconds() / 3600  # hours
            metrics['avg_time_between_tx_hours'] = time_diffs.mean()
            
            # Transaction frequency
            if metrics['activity_duration_days'] > 0:
                metrics['transaction_frequency_per_day'] = len(wallet_data) / metrics['activity_duration_days']
            else:
                metrics['transaction_frequency_per_day'] = len(wallet_data)
        
        return metrics
    
    def _calculate_risk_indicators(self, wallet_data: pd.DataFrame) -> Dict:
        """Calculate risk-related indicators."""
        metrics = {}
        
        # Liquidation flag
        metrics['was_liquidated'] = 1 if metrics.get('liquidationcall_count', 0) > 0 else 0
        
        # High borrowing intensity
        total_borrowed = metrics.get('total_borrow_usd', 0)
        total_deposited = metrics.get('total_deposit_usd', 0)
        
        if total_deposited > 0:
            metrics['borrow_to_deposit_ratio'] = total_borrowed / total_deposited
        else:
            metrics['borrow_to_deposit_ratio'] = 0 if total_borrowed == 0 else 10  # High ratio for borrowing without deposits
        
        # Asset concentration risk
        if metrics.get('unique_assets', 1) > 0:
            metrics['asset_concentration'] = 1 / metrics['unique_assets']
        else:
            metrics['asset_concentration'] = 1
        
        return metrics
    
    def _calculate_behavioral_features(self, wallet_data: pd.DataFrame) -> Dict:
        """Calculate advanced behavioral pattern features."""
        metrics = {}
        
        # Bot-like behavior indicators
        if len(wallet_data) > 5:
            # Very regular transaction timing might indicate bot behavior
            time_diffs = wallet_data['timestamp'].diff().dt.total_seconds()
            time_std = time_diffs.std()
            time_mean = time_diffs.mean()
            
            # Low coefficient of variation in timing suggests bot-like behavior
            if time_mean > 0:
                metrics['timing_regularity'] = time_std / time_mean
            else:
                metrics['timing_regularity'] = 0
        else:
            metrics['timing_regularity'] = 1
        
        # Repayment consistency
        repay_data = wallet_data[wallet_data['action'] == 'repay']
        if len(repay_data) > 0:
            metrics['repayment_consistency'] = len(repay_data) / len(wallet_data)
        else:
            metrics['repayment_consistency'] = 0
        
        # Deposit consistency
        deposit_data = wallet_data[wallet_data['action'] == 'deposit']
        if len(deposit_data) > 0:
            metrics['deposit_consistency'] = len(deposit_data) / len(wallet_data)
        else:
            metrics['deposit_consistency'] = 0
        
        return metrics
    
    def calculate_credit_score(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate credit scores using the weighted feature approach."""
        print("Calculating credit scores...")
        
        scores_df = features_df.copy()
        
        # Normalize features
        normalized_features = self._normalize_features(features_df)
        
        # Calculate weighted scores
        scores = []
        for idx, row in normalized_features.iterrows():
            score = self.base_score
            
            # Positive indicators
            score += self.feature_weights['repay_to_borrow_ratio'] * min(row['repay_to_borrow_ratio'], 1)
            score += self.feature_weights['deposit_consistency'] * row['deposit_consistency']
            score += self.feature_weights['repayment_frequency'] * row['repayment_consistency']
            score += self.feature_weights['portfolio_diversification'] * (1 - row['asset_concentration'])
            score += self.feature_weights['activity_duration_score'] * min(row['activity_duration_days'] / 365, 1)
            score += self.feature_weights['transaction_regularity'] * (1 - min(row['timing_regularity'], 1))
            
            # Negative indicators
            score += self.feature_weights['liquidation_penalty'] * row['was_liquidated']
            score += self.feature_weights['borrowing_intensity'] * min(row['borrow_to_deposit_ratio'] / 2, 1)
            score += self.feature_weights['high_risk_asset_exposure'] * row['asset_concentration']
            score += self.feature_weights['bot_like_behavior'] * (1 - min(row['timing_regularity'], 1))
            
            # Ensure score is within bounds
            score = max(self.min_score, min(self.max_score, score))
            scores.append(score)
        
        scores_df['credit_score'] = scores
        return scores_df
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features to 0-1 range for scoring."""
        normalized = features_df.copy()
        
        # Features to normalize
        numeric_features = [
            'repay_to_borrow_ratio', 'deposit_consistency', 'repayment_consistency',
            'asset_concentration', 'activity_duration_days', 'timing_regularity',
            'borrow_to_deposit_ratio', 'was_liquidated'
        ]
        
        for feature in numeric_features:
            if feature in normalized.columns:
                col_data = normalized[feature]
                if col_data.max() > col_data.min():
                    normalized[feature] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
                else:
                    normalized[feature] = 0
        
        return normalized
    
    def generate_analysis(self, scores_df: pd.DataFrame) -> None:
        """Generate analysis plots and statistics."""
        print("Generating analysis...")
        
        # Score distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(scores_df['credit_score'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Credit Score')
        plt.ylabel('Number of Wallets')
        plt.title('Credit Score Distribution')
        plt.grid(True, alpha=0.3)
        
        # Score ranges
        plt.subplot(2, 2, 2)
        score_ranges = pd.cut(scores_df['credit_score'], bins=[0, 200, 400, 600, 800, 1000], 
                             labels=['0-200', '201-400', '401-600', '601-800', '801-1000'])
        range_counts = score_ranges.value_counts().sort_index()
        plt.bar(range_counts.index, range_counts.values, color='lightcoral', alpha=0.7)
        plt.xlabel('Score Range')
        plt.ylabel('Number of Wallets')
        plt.title('Wallets by Score Range')
        plt.xticks(rotation=45)
        
        # Liquidation vs Score
        plt.subplot(2, 2, 3)
        liquidated = scores_df[scores_df['was_liquidated'] == 1]['credit_score']
        not_liquidated = scores_df[scores_df['was_liquidated'] == 0]['credit_score']
        
        plt.hist([liquidated, not_liquidated], bins=30, alpha=0.7, 
                label=['Liquidated', 'Not Liquidated'], color=['red', 'green'])
        plt.xlabel('Credit Score')
        plt.ylabel('Number of Wallets')
        plt.title('Score Distribution by Liquidation Status')
        plt.legend()
        
        # Feature correlation with score
        plt.subplot(2, 2, 4)
        correlation_features = ['repay_to_borrow_ratio', 'deposit_consistency', 'was_liquidated']
        correlations = [scores_df[['credit_score', feat]].corr().iloc[0, 1] for feat in correlation_features]
        
        plt.barh(correlation_features, correlations, color='orange', alpha=0.7)
        plt.xlabel('Correlation with Credit Score')
        plt.title('Feature Correlations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('credit_score_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("CREDIT SCORE ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total wallets analyzed: {len(scores_df)}")
        print(f"Average credit score: {scores_df['credit_score'].mean():.1f}")
        print(f"Median credit score: {scores_df['credit_score'].median():.1f}")
        print(f"Standard deviation: {scores_df['credit_score'].std():.1f}")
        
        print("\nScore distribution:")
        for range_name, count in range_counts.items():
            percentage = (count / len(scores_df)) * 100
            print(f"  {range_name}: {count} wallets ({percentage:.1f}%)")
        
        # Low score analysis
        low_score_wallets = scores_df[scores_df['credit_score'] <= 300]
        print(f"\nLow score wallets (≤300): {len(low_score_wallets)}")
        if len(low_score_wallets) > 0:
            print(f"  Average liquidation rate: {low_score_wallets['was_liquidated'].mean():.1%}")
            print(f"  Average repay-to-borrow ratio: {low_score_wallets['repay_to_borrow_ratio'].mean():.2f}")
        
        # High score analysis
        high_score_wallets = scores_df[scores_df['credit_score'] >= 700]
        print(f"\nHigh score wallets (≥700): {len(high_score_wallets)}")
        if len(high_score_wallets) > 0:
            print(f"  Average liquidation rate: {high_score_wallets['was_liquidated'].mean():.1%}")
            print(f"  Average repay-to-borrow ratio: {high_score_wallets['repay_to_borrow_ratio'].mean():.2f}")


def main():
    """Main execution function."""
    # Initialize the credit scorer
    scorer = AaveCreditScorer()
    
    # Load and process data
    try:
        df = scorer.load_data('user-wallet-transactions.json')
    except FileNotFoundError:
        print("Error: user-wallet-transactions.json file not found!")
        print("Please ensure the file is in the same directory as this script.")
        return
    
    # Engineer features
    features_df = scorer.engineer_features(df)
    
    # Calculate credit scores
    scores_df = scorer.calculate_credit_score(features_df)
    
    # Generate analysis
    scorer.generate_analysis(scores_df)
    
    # Save results
    output_columns = ['userWallet', 'credit_score', 'total_transactions', 
                     'repay_to_borrow_ratio', 'was_liquidated', 'unique_assets']
    
    output_df = scores_df[output_columns].sort_values('credit_score', ascending=False)
    output_df.to_csv('wallet_credit_scores.csv', index=False)
    
    print(f"\nResults saved to 'wallet_credit_scores.csv'")
    print(f"Analysis plot saved to 'credit_score_analysis.png'")
    
    # Display top and bottom wallets
    print("\n" + "="*50)
    print("TOP 10 HIGHEST SCORING WALLETS")
    print("="*50)
    print(output_df.head(10).to_string(index=False))
    
    print("\n" + "="*50)
    print("TOP 10 LOWEST SCORING WALLETS")
    print("="*50)
    print(output_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()