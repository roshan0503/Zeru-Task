#!/usr/bin/env python3
"""
Data Exploration Script for Aave V2 Transactions
================================================

This script performs initial data exploration and validation
to understand the structure and quality of the transaction data.

Usage:
    python data_explorer.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(filepath='user-wallet-transactions.json'):
    """Load and perform initial exploration of the transaction data."""
    
    print("Loading transaction data...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        print(f"✓ Successfully loaded {len(df)} transactions")
        
    except FileNotFoundError:
        print(f"✗ Error: {filepath} not found!")
        return None
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON format in {filepath}")
        return None
    
    print("\n" + "="*60)
    print("BASIC DATA STRUCTURE")
    print("="*60)
    
    # Basic info
    print(f"Total transactions: {len(df):,}")
    print(f"Unique wallets: {df['userWallet'].nunique():,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Data types:\n{df.dtypes}")
    
    # Column overview
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    
    # Missing values
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    return df

def explore_actions(df):
    """Analyze action types and their distributions."""
    
    print("\n" + "="*60)
    print("ACTION ANALYSIS")
    print("="*60)
    
    # Action distribution
    action_counts = df['action'].value_counts()
    print("Action type distribution:")
    for action, count in action_counts.items():
        percentage = count / len(df) * 100
        print(f"  - {action}: {count:,} ({percentage:.1f}%)")
    
    # Action by wallet analysis
    wallet_actions = df.groupby('userWallet')['action'].value_counts().unstack(fill_value=0)
    
    print(f"\nWallet behavior patterns:")
    print(f"  - Wallets with only deposits: {(wallet_actions['deposit'] > 0).sum() & (wallet_actions.drop('deposit', axis=1).sum(axis=1) == 0).sum()}")
    print(f"  - Wallets with borrowing: {(wallet_actions['borrow'] > 0).sum()}")
    print(f"  - Wallets with repayments: {(wallet_actions['repay'] > 0).sum()}")
    print(f"  - Wallets with liquidations: {(wallet_actions['liquidationcall'] > 0).sum()}")
    
    # Visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    action_counts.plot(kind='bar', color='skyblue')
    plt.title('Transaction Types Distribution')
    plt.xlabel('Action Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    transactions_per_wallet = df.groupby('userWallet').size()
    plt.hist(transactions_per_wallet, bins=50, alpha=0.7, color='lightcoral')
    plt.title('Transactions per Wallet Distribution')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Number of Wallets')
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    # Timeline of actions
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    daily_actions = df.groupby(['date', 'action']).size().unstack(fill_value=0)
    daily_actions.plot(kind='area', stacked=True, alpha=0.7)
    plt.title('Daily Transaction Volume by Type')
    plt.xlabel('Date')
    plt.ylabel('Transaction Count')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.subplot(2, 2, 4)
    # Repay to borrow ratio distribution
    wallet_ratios = []
    for wallet in df['userWallet'].unique():
        wallet_data = df[df['userWallet'] == wallet]
        borrows = len(wallet_data[wallet_data['action'] == 'borrow'])
        repays = len(wallet_data[wallet_data['action'] == 'repay'])
        if borrows > 0:
            ratio = repays / borrows
            wallet_ratios.append(min(ratio, 3))  # Cap at 3 for visualization
    
    plt.hist(wallet_ratios, bins=50, alpha=0.7, color='lightgreen')
    plt.title('Repay-to-Borrow Ratio Distribution')
    plt.xlabel('Ratio (capped at 3)')
    plt.ylabel('Number of Wallets')
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    plt.show()

def explore_financial_data(df):
    """Analyze financial aspects of transactions."""
    
    print("\n" + "="*60)
    print("FINANCIAL DATA ANALYSIS")
    print("="*60)
    
    # Extract action data
    action_data = pd.json_normalize(df['actionData'])
    df_financial = pd.concat([df, action_data], axis=1)
    
    # Convert to numeric
    df_financial['amount'] = pd.to_numeric(df_financial['amount'], errors='coerce')
    df_financial['assetPriceUSD'] = pd.to_numeric(df_financial['assetPriceUSD'], errors='coerce')
    df_financial['amountUSD'] = df_financial['amount'] * df_financial['assetPriceUSD']
    
    # Clean data
    df_financial = df_financial.dropna(subset=['amount', 'assetPriceUSD'])
    df_financial = df_financial[df_financial['amountUSD'] > 0]
    
    print(f"Financial transactions analyzed: {len(df_financial):,}")
    print(f"Total volume (USD): ${df_financial['amountUSD'].sum():,.2f}")
    print(f"Average transaction size: ${df_financial['amountUSD'].mean():,.2f}")
    print(f"Median transaction size: ${df_financial['amountUSD'].median():,.2f}")
    
    # Asset analysis
    print(f"\nAsset distribution:")
    asset_counts = df_financial['assetSymbol'].value_counts()
    for asset, count in asset_counts.head(10).items():
        volume = df_financial[df_financial['assetSymbol'] == asset]['amountUSD'].sum()
        print(f"  - {asset}: {count:,} transactions, ${volume:,.2f} volume")
    
    # Volume by action
    print(f"\nVolume by action type:")
    action_volumes = df_financial.groupby('action')['amountUSD'].sum()
    for action, volume in action_volumes.items():
        print(f"  - {action}: ${volume:,.2f}")
    
    return df_financial

def analyze_wallet_patterns(df):
    """Analyze individual wallet behavior patterns."""
    
    print("\n" + "="*60)
    print("WALLET PATTERN ANALYSIS")
    print("="*60)
    
    wallet_stats = []
    
    for wallet in df['userWallet'].unique():
        wallet_data = df[df['userWallet'] == wallet]
        
        stats = {
            'wallet': wallet,
            'total_transactions': len(wallet_data),
            'unique_actions': wallet_data['action'].nunique(),
            'first_transaction': wallet_data['timestamp'].min(),
            'last_transaction': wallet_data['timestamp'].max(),
            'activity_days': (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()) / (24*3600),
            'actions': wallet_data['action'].value_counts().to_dict()
        }
        
        wallet_stats.append(stats)
    
    wallet_df = pd.DataFrame(wallet_stats)
    
    print(f"Wallet activity patterns:")
    print(f"  - Average transactions per wallet: {wallet_df['total_transactions'].mean():.1f}")
    print(f"  - Median transactions per wallet: {wallet_df['total_transactions'].median():.1f}")
    print(f"  - Most active wallet: {wallet_df['total_transactions'].max()} transactions")
    print(f"  - Average activity duration: {wallet_df['activity_days'].mean():.1f} days")
    
    # Identify interesting patterns
    single_action_wallets = wallet_df[wallet_df['unique_actions'] == 1]
    print(f"  - Single-action wallets: {len(single_action_wallets)}")
    
    high_activity_wallets = wallet_df[wallet_df['total_transactions'] > 100]
    print(f"  - High-activity wallets (>100 tx): {len(high_activity_wallets)}")
    
    return wallet_df

def generate_data_quality_report(df):
    """Generate a comprehensive data quality report."""
    
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    # Timestamp analysis
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')
    print(f"Timestamp range: {df['timestamp_dt'].min()} to {df['timestamp_dt'].max()}")
    
    # Duplicate analysis
    duplicate_txhash = df['txHash'].duplicated().sum()
    print(f"Duplicate transaction hashes: {duplicate_txhash}")
    
    # Action data completeness
    action_data_fields = ['amount', 'assetSymbol', 'assetPriceUSD']
    print(f"\nAction data completeness:")
    
    for field in action_data_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            print(f"  - {field}: {non_null:,}/{len(df):,} ({non_null/len(df)*100:.1f}%)")
    
    # Validation checks
    print(f"\nValidation checks:")
    print(f"  - Valid userWallet addresses: {df['userWallet'].str.startswith('0x').sum()}")
    print(f"  - Valid transaction hashes: {df['txHash'].str.len().eq(66).sum()}")
    print(f"  - Future timestamps: {(df['timestamp_dt'] > datetime.now()).sum()}")
    
    return df

def main():
    """Main exploration function."""
    
    print("🔍 Starting Aave V2 Transaction Data Exploration")
    print("=" * 80)
    
    # Load and explore basic structure
    df = load_and_explore_data()
    if df is None:
        return
    
    # Explore actions
    explore_actions(df)
    
    # Explore financial data
    df_financial = explore_financial_data(df)
    
    # Analyze wallet patterns
    wallet_df = analyze_wallet_patterns(df)
    
    # Generate quality report
    df_clean = generate_data_quality_report(df)
    
    print("\n" + "="*80)
    print("✅ Data exploration complete!")
    print("📊 Visualization saved as 'data_exploration.png'")
    print("📋 Run 'python generate_scores.py' to generate credit scores")
    print("=" * 80)

if __name__ == "__main__":
    main()