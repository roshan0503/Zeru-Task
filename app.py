import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, Counter
import warnings
from typing import Dict

# Suppress warnings for cleaner output in Streamlit
warnings.filterwarnings('ignore')

# Set Streamlit page configuration for a wider layout and custom title
st.set_page_config(layout="wide", page_title="Aave V2 Wallet Analytics & Credit Scoring", page_icon="📊")

# --- AaveCreditScorer Class (Adapted from generate_scores.py) ---
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

    @st.cache_data
    def load_data(_self, uploaded_file) -> pd.DataFrame:
        """Load and preprocess transaction data from an uploaded file."""
        st.info("Loading transaction data...")
        try:
            data = json.load(uploaded_file)
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

            st.success(f"Loaded {len(df)} transactions from {df['userWallet'].nunique()} unique wallets.")
            return df
        except json.JSONDecodeError:
            st.error("Error: Invalid JSON format. Please upload a valid JSON file.")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"An error occurred while loading data: {e}")
            return pd.DataFrame()

    @st.cache_data
    def engineer_features(_self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for each wallet based on transaction history."""
        st.info("Engineering features for each wallet...")

        features = []
        unique_wallets = df['userWallet'].unique()
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, wallet in enumerate(unique_wallets):
            wallet_data = df[df['userWallet'] == wallet].copy()
            wallet_data = wallet_data.sort_values('timestamp')

            feature_dict = {'userWallet': wallet}

            # Basic transaction metrics
            feature_dict.update(_self._calculate_transaction_metrics(wallet_data))

            # Financial metrics
            feature_dict.update(_self._calculate_financial_metrics(wallet_data))

            # Time-based metrics
            feature_dict.update(_self._calculate_time_metrics(wallet_data))

            # Risk indicators
            feature_dict.update(_self._calculate_risk_indicators(wallet_data))

            # Advanced behavioral features
            feature_dict.update(_self._calculate_behavioral_features(wallet_data))

            features.append(feature_dict)

            # Update progress bar
            progress = (i + 1) / len(unique_wallets)
            progress_bar.progress(progress)
            status_text.text(f"Processing wallet {i+1}/{len(unique_wallets)}...")

        st.success("Features engineered successfully!")
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
        metrics['was_liquidated'] = 1 if wallet_data['action'].str.contains('liquidationcall').any() else 0

        # High borrowing intensity
        total_borrowed = wallet_data[wallet_data['action'] == 'borrow']['amountUSD'].sum()
        total_deposited = wallet_data[wallet_data['action'] == 'deposit']['amountUSD'].sum()

        if total_deposited > 0:
            metrics['borrow_to_deposit_ratio'] = total_borrowed / total_deposited
        else:
            metrics['borrow_to_deposit_ratio'] = 0 if total_borrowed == 0 else 10  # High ratio for borrowing without deposits

        # Asset concentration risk
        unique_assets = wallet_data['assetSymbol'].nunique()
        if unique_assets > 0:
           metrics['asset_concentration'] = 1 / unique_assets
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

    @st.cache_data
    def calculate_credit_score(_self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate credit scores using the weighted feature approach."""
        st.info("Calculating credit scores...")

        scores_df = features_df.copy()

        # Normalize features
        normalized_features = _self._normalize_features(features_df)

        # Calculate weighted scores
        scores = []
        for idx, row in normalized_features.iterrows():
            score = _self.base_score

            # Positive indicators
            score += _self.feature_weights['repay_to_borrow_ratio'] * min(row['repay_to_borrow_ratio'], 1)
            score += _self.feature_weights['deposit_consistency'] * row['deposit_consistency']
            score += _self.feature_weights['repayment_frequency'] * row['repayment_consistency']
            score += _self.feature_weights['portfolio_diversification'] * (1 - row['asset_concentration'])
            score += _self.feature_weights['activity_duration_score'] * min(row['activity_duration_days'] / 365, 1)
            score += _self.feature_weights['transaction_regularity'] * (1 - min(row['timing_regularity'], 1))

            # Negative indicators
            score += _self.feature_weights['liquidation_penalty'] * row['was_liquidated']
            score += _self.feature_weights['borrowing_intensity'] * min(row['borrow_to_deposit_ratio'] / 2, 1)
            score += _self.feature_weights['high_risk_asset_exposure'] * row['asset_concentration']
            score += _self.feature_weights['bot_like_behavior'] * (min(row['timing_regularity'], 1)) # Higher regularity means more bot-like

            # Ensure score is within bounds
            score = max(_self.min_score, min(_self.max_score, score))
            scores.append(score)

        scores_df['credit_score'] = scores
        st.success("Credit scores calculated!")
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
                    normalized[feature] = 0 # If all values are same, normalize to 0

        return normalized

    def generate_analysis(self, scores_df: pd.DataFrame) -> None:
        """Generate analysis plots and statistics."""
        st.subheader("Credit Score Analysis Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Wallets Analyzed", len(scores_df))
        col2.metric("Average Credit Score", f"{scores_df['credit_score'].mean():.1f}")
        col3.metric("Median Credit Score", f"{scores_df['credit_score'].median():.1f}")

        # Score distribution
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(scores_df['credit_score'], bins=50, kde=True, color='skyblue', ax=ax1)
        ax1.set_xlabel('Credit Score')
        ax1.set_ylabel('Number of Wallets')
        ax1.set_title('Credit Score Distribution')
        st.pyplot(fig1)

        # Score ranges
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        score_ranges = pd.cut(scores_df['credit_score'], bins=[0, 200, 400, 600, 800, 1000],
                             labels=['0-200', '201-400', '401-600', '601-800', '801-1000'])
        range_counts = score_ranges.value_counts().sort_index()
        sns.barplot(x=range_counts.index, y=range_counts.values, palette='viridis', ax=ax2)
        ax2.set_xlabel('Score Range')
        ax2.set_ylabel('Number of Wallets')
        ax2.set_title('Wallets by Score Range')
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

        # Liquidation vs Score
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        liquidated = scores_df[scores_df['was_liquidated'] == 1]['credit_score']
        not_liquidated = scores_df[scores_df['was_liquidated'] == 0]['credit_score']

        sns.histplot(liquidated, bins=30, alpha=0.7, label='Liquidated', color='red', kde=True, ax=ax3)
        sns.histplot(not_liquidated, bins=30, alpha=0.7, label='Not Liquidated', color='green', kde=True, ax=ax3)
        ax3.set_xlabel('Credit Score')
        ax3.set_ylabel('Number of Wallets')
        ax3.set_title('Score Distribution by Liquidation Status')
        ax3.legend()
        st.pyplot(fig3)

        # Feature correlation with score
        st.subheader("Key Feature Correlations with Credit Score")
        correlation_features = [
            'repay_to_borrow_ratio', 'deposit_consistency', 'was_liquidated',
            'activity_duration_days', 'unique_assets', 'total_transactions',
            'borrow_to_deposit_ratio', 'timing_regularity'
        ]
        
        # Filter for features that exist in the dataframe
        existing_correlation_features = [f for f in correlation_features if f in scores_df.columns]
        
        if existing_correlation_features:
            correlations = scores_df[['credit_score'] + existing_correlation_features].corr().iloc[0, 1:]
            
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=correlations.values, y=correlations.index, palette='coolwarm', ax=ax4)
            ax4.set_xlabel('Correlation with Credit Score')
            ax4.set_title('Feature Correlations')
            ax4.grid(axis='x', alpha=0.3)
            st.pyplot(fig4)
        else:
            st.warning("No relevant features found for correlation analysis.")


# --- Data Exploration Functions (Adapted from data_explorer.py) ---
def display_basic_data_structure(df: pd.DataFrame):
    """Display basic info about the DataFrame."""
    st.subheader("Basic Data Structure Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Unique Wallets", f"{df['userWallet'].nunique():,}")
    
    if not df.empty and 'timestamp' in df.columns:
        col3.metric("Date Range", f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}")
    else:
        col3.metric("Date Range", "N/A")

    st.write("### Data Types")
    # Convert dtypes to string to avoid PyArrow serialization issues
    st.dataframe(df.dtypes.astype(str).reset_index().rename(columns={'index': 'Column', 0: 'Data Type'}))

    st.write("### Missing Values")
    missing = df.isnull().sum()
    missing_df = missing[missing > 0].reset_index().rename(columns={'index': 'Column', 0: 'Missing Count'})
    missing_df['Percentage'] = (missing_df['Missing Count'] / len(df) * 100).round(1)
    if not missing_df.empty:
        # Convert 'Data Type' column to string if it exists in missing_df (though it usually won't for missing values)
        if 'Data Type' in missing_df.columns:
            missing_df['Data Type'] = missing_df['Data Type'].astype(str)
        st.dataframe(missing_df)
    else:
        st.info("No missing values found in the dataset.")

def display_action_analysis_graphs(df: pd.DataFrame):
    """Display action analysis graphs."""
    st.subheader("Transaction Action Analysis Graphs")

    action_counts = df['action'].value_counts()
    
    st.write("#### 1. Action Type Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=action_counts.index, y=action_counts.values, palette='pastel', ax=ax1)
    ax1.set_title('Transaction Types Distribution')
    ax1.set_xlabel('Action Type')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.write("#### 2. Transactions per Wallet Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    transactions_per_wallet = df.groupby('userWallet').size()
    sns.histplot(transactions_per_wallet, bins=50, kde=True, color='lightcoral', ax=ax2)
    ax2.set_title('Transactions per Wallet Distribution')
    ax2.set_xlabel('Number of Transactions')
    ax2.set_ylabel('Number of Wallets (Log Scale)')
    ax2.set_yscale('log')
    st.pyplot(fig2)

    st.write("#### 3. Daily Transaction Volume by Type")
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_actions = df.groupby(['date', 'action']).size().unstack(fill_value=0)
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    daily_actions.plot(kind='area', stacked=True, alpha=0.7, ax=ax3)
    ax3.set_title('Daily Transaction Volume by Type')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Transaction Count')
    ax3.legend(title='Action Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig3)

    st.write("#### 4. Repay-to-Borrow Ratio Distribution")
    wallet_ratios = []
    for wallet in df['userWallet'].unique():
        wallet_data = df[df['userWallet'] == wallet]
        borrows = len(wallet_data[wallet_data['action'] == 'borrow'])
        repays = len(wallet_data[wallet_data['action'] == 'repay'])
        if borrows > 0:
            ratio = repays / borrows
            wallet_ratios.append(min(ratio, 3))  # Cap at 3 for visualization
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.histplot(wallet_ratios, bins=50, kde=True, color='lightgreen', ax=ax4)
    ax4.set_title('Repay-to-Borrow Ratio Distribution')
    ax4.set_xlabel('Ratio (capped at 3)')
    ax4.set_ylabel('Number of Wallets')
    st.pyplot(fig4)

def display_financial_data_graphs(df: pd.DataFrame):
    """Display financial data analysis graphs."""
    st.subheader("Financial Data Analysis Graphs")

    # Ensure amount and assetPriceUSD are numeric and not null
    df_financial = df.copy()
    df_financial['amount'] = pd.to_numeric(df_financial['amount'], errors='coerce')
    df_financial['assetPriceUSD'] = pd.to_numeric(df_financial['assetPriceUSD'], errors='coerce')
    df_financial['amountUSD'] = df_financial['amount'] * df_financial['assetPriceUSD']

    df_financial = df_financial.dropna(subset=['amount', 'assetPriceUSD', 'amountUSD'])
    df_financial = df_financial[df_financial['amountUSD'] > 0]

    if df_financial.empty:
        st.warning("No valid financial transactions found after cleaning to generate graphs.")
        return

    st.write("#### 1. Top 10 Asset Distribution by Volume")
    asset_volume = df_financial.groupby('assetSymbol')['amountUSD'].sum().sort_values(ascending=False).head(10)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=asset_volume.index, y=asset_volume.values, palette='magma', ax=ax1)
    ax1.set_title('Top 10 Asset Distribution by Volume (USD)')
    ax1.set_xlabel('Asset Symbol')
    ax1.set_ylabel('Total Volume (USD)')
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    st.write("#### 2. Volume by Action Type")
    action_volumes = df_financial.groupby('action')['amountUSD'].sum().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=action_volumes.index, y=action_volumes.values, palette='cividis', ax=ax2)
    ax2.set_title('Total Volume by Action Type (USD)')
    ax2.set_xlabel('Action Type')
    ax2.set_ylabel('Total Volume (USD)')
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

def display_wallet_pattern_graphs(df: pd.DataFrame):
    """Display wallet pattern analysis graphs."""
    st.subheader("Wallet Pattern Analysis Graphs")

    wallet_stats = []
    for wallet in df['userWallet'].unique():
        wallet_data = df[df['userWallet'] == wallet]
        
        stats = {
            'wallet': wallet,
            'total_transactions': len(wallet_data),
            'unique_actions': wallet_data['action'].nunique(),
            'first_transaction': wallet_data['timestamp'].min(),
            'last_transaction': wallet_data['timestamp'].max(),
            'activity_days': (wallet_data['timestamp'].max() - wallet_data['timestamp'].min()).total_seconds() / (24*3600) if len(wallet_data) > 1 else 0,
            'actions': wallet_data['action'].value_counts().to_dict()
        }
        wallet_stats.append(stats)
    
    wallet_df = pd.DataFrame(wallet_stats)

    st.write("#### 1. Wallet Activity Duration Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(wallet_df['activity_days'], bins=50, kde=True, color='purple', ax=ax1)
    ax1.set_title('Wallet Activity Duration Distribution (Days)')
    ax1.set_xlabel('Activity Duration (Days)')
    ax1.set_ylabel('Number of Wallets')
    st.pyplot(fig1)

    st.write("#### 2. Wallets by Unique Actions")
    unique_action_counts = wallet_df['unique_actions'].value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(x=unique_action_counts.index, y=unique_action_counts.values, palette='cubehelix', ax=ax2)
    ax2.set_title('Number of Wallets by Unique Action Types')
    ax2.set_xlabel('Number of Unique Actions')
    ax2.set_ylabel('Number of Wallets')
    st.pyplot(fig2)


def display_data_quality_report(df: pd.DataFrame):
    """Generate a comprehensive data quality report."""
    st.subheader("Data Quality Report")

    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')

    st.write(f"- Timestamp range: **{df['timestamp_dt'].min()}** to **{df['timestamp_dt'].max()}**")

    duplicate_txhash = df['txHash'].duplicated().sum()
    st.write(f"- Duplicate transaction hashes: **{duplicate_txhash:,}**")

    st.write("#### Action Data Completeness")
    action_data_fields = ['amount', 'assetSymbol', 'assetPriceUSD']
    completeness_data = []
    for field in action_data_fields:
        if field in df.columns:
            non_null = df[field].notna().sum()
            completeness_data.append({
                'Field': field,
                'Non-Null Count': f"{non_null:,}/{len(df):,}",
                'Completeness (%)': f"{non_null/len(df)*100:.1f}%"
            })
    st.dataframe(pd.DataFrame(completeness_data))

    st.write("#### Validation Checks")
    st.write(f"- Valid `userWallet` addresses (start with '0x'): **{df['userWallet'].astype(str).str.startswith('0x').sum():,}**")
    st.write(f"- Valid `txHash` lengths (66 characters): **{df['txHash'].astype(str).str.len().eq(66).sum():,}**")
    st.write(f"- Future timestamps: **{(df['timestamp_dt'] > datetime.now()).sum():,}**")


# --- Main Streamlit Application ---
def main():
    st.title("📊 Aave V2 Wallet Analytics & Credit Scoring")
    st.markdown("""
        Upload your `user-wallet-transactions.json` file to explore transaction data
        and generate credit scores for Aave V2 wallet addresses.
    """)

    uploaded_file = st.file_uploader("Upload JSON Dataset", type="json")

    if uploaded_file is not None:
        scorer = AaveCreditScorer()
        
        # Load data
        df = scorer.load_data(uploaded_file)

        if not df.empty:
            st.success("Dataset loaded successfully!")

            # Create tabs for different sections
            tab1, tab2 = st.tabs(["🔍 Data Exploration", "💯 Credit Scoring"])

            with tab1:
                st.title("Data Exploration") # Changed to st.title for larger text
                st.markdown("Dive into the raw transaction data to understand its structure and patterns.")
                
                display_basic_data_structure(df)
                st.markdown("---")

                graph_options = {
                    "Select a graph category": None,
                    "Action Analysis Graphs": "action_analysis",
                    "Financial Data Analysis Graphs": "financial_data",
                    "Wallet Pattern Analysis Graphs": "wallet_pattern",
                }
                
                selected_graph_category = st.selectbox(
                    "Choose a graph category to display:",
                    list(graph_options.keys())
                )

                if graph_options[selected_graph_category] == "action_analysis":
                    display_action_analysis_graphs(df)
                elif graph_options[selected_graph_category] == "financial_data":
                    display_financial_data_graphs(df)
                elif graph_options[selected_graph_category] == "wallet_pattern":
                    display_wallet_pattern_graphs(df)
                else:
                    st.info("Select a graph category from the dropdown above to view the visualizations.")

                st.markdown("---")
                display_data_quality_report(df)

            with tab2:
                st.title("Aave V2 Credit Scoring") # Changed to st.title for larger text
                st.markdown("Generate and analyze credit scores based on wallet transaction behavior.")

                if st.button("Generate Credit Scores"):
                    with st.spinner("Processing data and calculating scores... This may take a while for large datasets."):
                        features_df = scorer.engineer_features(df)
                        if not features_df.empty:
                            scores_df = scorer.calculate_credit_score(features_df)
                            
                            if not scores_df.empty:
                                scorer.generate_analysis(scores_df)

                                st.subheader("Top & Bottom Scoring Wallets")
                                output_columns = ['userWallet', 'credit_score', 'total_transactions',
                                                 'repay_to_borrow_ratio', 'was_liquidated', 'unique_assets']
                                
                                # Filter columns that actually exist in scores_df
                                existing_output_columns = [col for col in output_columns if col in scores_df.columns]

                                output_df = scores_df[existing_output_columns].sort_values('credit_score', ascending=False)

                                st.write("#### Top 10 Highest Scoring Wallets")
                                st.dataframe(output_df.head(10))

                                st.write("#### Top 10 Lowest Scoring Wallets")
                                st.dataframe(output_df.tail(10))

                                st.download_button(
                                    label="Download Credit Scores as CSV",
                                    data=output_df.to_csv(index=False).encode('utf-8'),
                                    file_name='wallet_credit_scores.csv',
                                    mime='text/csv',
                                )
                            else:
                                st.error("Could not calculate credit scores. Please check the data.")
                        else:
                            st.error("Could not engineer features. Please check the data.")
        else:
            st.warning("Please upload a valid JSON file to proceed.")

if __name__ == "__main__":
    main()
