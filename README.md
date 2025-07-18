# Aave V2 Wallet Analytics & Credit Scoring

## Project Overview

This project develops a robust machine learning model to assign a credit score between 0 and 1000 to Aave V2 protocol wallets. The scoring is based solely on historical transaction behavior, aiming to identify reliable and responsible users (higher scores) versus risky, bot-like, or exploitative behavior (lower scores). The solution includes a Streamlit application for interactive data exploration, feature engineering, credit score calculation, and comprehensive analysis.

## Problem Statement

The core challenge is to analyze raw, transaction-level data from the Aave V2 protocol, where each record represents a wallet's interaction (deposit, borrow, repay, redeemunderlying, liquidationcall). From this data, the goal is to:

- Engineer meaningful features from DeFi transaction behavior
- Implement a one-step script (provided as `app.py` and run via Streamlit) that generates wallet scores from a JSON file containing user transactions
- Validate and explain the score logic for transparency and extensibility
- Provide an analysis of the scored wallets, including distribution graphs and behavioral insights

## Methodology and Architecture

The credit scoring system is built around a weighted combination of engineered features, reflecting both responsible and risky behaviors in DeFi lending.

### 1. Data Loading and Preprocessing

**Input:** A JSON file containing raw transaction data from Aave V2.

**Process:**
- The `load_data` function reads the JSON file and converts it into a pandas DataFrame
- `timestamp` values are converted from Unix timestamps to datetime objects for easier time-series analysis
- Nested `actionData` (containing amount, assetSymbol, assetPriceUSD, etc.) is normalized and merged into the main DataFrame
- `amount` and `assetPriceUSD` are converted to numeric types, with missing values handled (coerced to 0)
- `amountUSD` is calculated as `amount * assetPriceUSD` to standardize transaction values

### 2. Feature Engineering

The `engineer_features` function extracts a comprehensive set of features for each unique wallet, categorized as follows:

**Basic Transaction Metrics:**
- `total_transactions`: Total number of transactions
- `deposit_count`, `borrow_count`, `repay_count`, `redeemunderlying_count`, `liquidationcall_count`: Counts of specific action types
- `repay_to_borrow_ratio`: Ratio of repay actions to borrow actions (indicator of responsibility)
- `liquidation_to_total_ratio`: Ratio of liquidation calls to total transactions (indicator of risk)

**Financial Metrics:**
- `total_deposit_usd`, `total_borrow_usd`, `total_repay_usd`, `total_redeemunderlying_usd`: Total USD volume for each action
- `avg_transaction_usd`, `max_transaction_usd`: Average and maximum transaction sizes
- `unique_assets`: Number of distinct assets interacted with (indicator of portfolio diversification)
- `avg_borrow_rate`, `max_borrow_rate`: Average and maximum borrow rates (if applicable)

**Time-Based Metrics:**
- `activity_duration_days`: Total duration of wallet activity in days
- `avg_time_between_tx_hours`: Average time difference between consecutive transactions
- `transaction_frequency_per_day`: Number of transactions per day

**Risk Indicators:**
- `was_liquidated`: Binary flag indicating if the wallet experienced a liquidation call
- `borrow_to_deposit_ratio`: Ratio of total borrowed USD to total deposited USD (indicator of leverage/risk)
- `asset_concentration`: Inverse of unique assets (higher value means less diversification)

**Advanced Behavioral Features:**
- `timing_regularity`: Coefficient of variation of time differences between transactions (lower value suggests bot-like behavior)
- `repayment_consistency`: Ratio of repay actions to total actions
- `deposit_consistency`: Ratio of deposit actions to total actions

### 3. Credit Scoring Logic

The `calculate_credit_score` function assigns a score between 0 and 1000 using a weighted sum approach:

- **Base Score:** Each wallet starts with a `base_score` of 500
- **Feature Normalization:** All relevant features are normalized to a 0-1 range to ensure fair contribution regardless of their original scale
- **Weighted Combination:** Each normalized feature is multiplied by a predefined `feature_weights` value and added to (or subtracted from) the base score

**Positive Indicators:** Features like `repay_to_borrow_ratio`, `deposit_consistency`, `repayment_consistency`, `portfolio_diversification`, `activity_duration_score`, and `transaction_regularity` (inverse of timing_regularity) contribute positively.

**Negative Indicators:** Features like `liquidation_penalty` (for was_liquidated), `borrowing_intensity` (for borrow_to_deposit_ratio), `high_risk_asset_exposure` (for asset_concentration), and `bot_like_behavior` (for timing_regularity) contribute negatively.

- **Score Clamping:** The final score is clamped between `min_score` (0) and `max_score` (1000)

### 4. Score Validation and Explanation

The `generate_analysis` function provides visual and statistical summaries of the calculated credit scores, including:

- Overall score distribution (histogram)
- Distribution of wallets across predefined score ranges (e.g., 0-200, 201-400)
- Comparison of score distributions for liquidated vs. non-liquidated wallets
- Correlation of key features with the final credit score, offering insights into feature impact

## Processing Flow

1. **User Uploads JSON:** The user uploads the `user-wallet-transactions.json` file via the Streamlit interface
2. **Data Loading:** The `AaveCreditScorer.load_data()` method reads and preprocesses the JSON
3. **Feature Engineering:** The `AaveCreditScorer.engineer_features()` method calculates various behavioral and financial features for each unique wallet
4. **Credit Score Calculation:** The `AaveCreditScorer.calculate_credit_score()` method uses the engineered features and predefined weights to compute a credit score for each wallet
5. **Analysis Generation:** The `AaveCreditScorer.generate_analysis()` method generates and displays interactive plots and summary statistics of the credit scores
6. **Results Display & Download:** The Streamlit app displays top/bottom scoring wallets and provides a download button for the full credit scores CSV

## Deliverables

- **Github Repository:** Containing all source code
- **README.md:** This file, explaining the methodology, architecture, and processing flow
- **analysis.md:** A detailed analysis of the scored wallets, including score distribution graphs and insights into wallet behavior across different score ranges
- **app.py:** The Streamlit application script
- **wallet_credit_scores.csv:** Sample output of the credit scoring process

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/aave-credit-scoring.git
   cd aave-credit-scoring
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Create a `requirements.txt` file with streamlit, pandas, numpy, matplotlib, seaborn)

4. **Download the dataset:**
   Download the `user-transactions.json` file from one of the following links:
   - [Raw JSON (~87MB)](https://drive.google.com/file/d/1ISFbAXxadMrt7Zl96rmzzZmEKZnyW7FS/view?usp=sharing)
   - [Compressed ZIP (~10MB)](https://drive.google.com/file/d/14ceBCLQ-BTcydDrFJauVA_PKAZ7VtDor/view?usp=sharing)
   
   Place this file in the same directory as `app.py` or be ready to upload it via the Streamlit interface.

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
   
   This will open the application in your web browser. Upload the downloaded JSON file to proceed with the analysis and scoring.

## Dependencies

- streamlit
- pandas
- numpy
- json (built-in)
- matplotlib
- seaborn
- datetime (built-in)
- collections (built-in)
