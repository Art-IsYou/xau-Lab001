# XAU/USD Data Pipeline & Analysis

Repository for processing, validating, and analyzing minute-resolution XAU/USD price data from MT5.

## Project Structure

```
xau-Lab-001/
├── data/
│   ├── load.py        # CSV loading and UTC timezone normalization
│   ├── validate.py    # Data validation and cleaning
│   └── resample.py    # Timezone-aware bar aggregation
├── signals/           # Future signal generation module
├── tests/             # Pytest test suite
├── explore.ipynb      # Jupyter notebook for exploration
└── analyze.py         # Command-line analysis script
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Load and Validate Data

Use the command-line tool to process CSV files.

**Auto-detect timezone (recommended):**
```bash
python -m data.load /path/to/your/data.csv
```

**Specify timezone (if known):**
```bash
# UTC
python -m data.load /path/to/data.csv UTC

# Europe/Athens (UTC+2 in winter, UTC+3 in summer)
python -m data.load /path/to/data.csv Europe/Athens

# Fixed offset (use with caution)
python -m data.load /path/to/data.csv "+2"
```

### 2. Check Data Integrity

Validate cleaned data for gaps, spikes, and quality issues.

```bash
python -m data.check_data /path/to/your/cleaned.csv
```

**Example output:**
```
loaded: 898765 rows | 2024-01-01 00:01:00+02:00 -> 2024-12-31 23:59:00+02:00
validation report summary:
rows: 898765 in -> 898120 out (645 removed)
  unsorted input: False (sorted)
  duplicate timestamps removed: 210
  NaN rows removed: 1
  non-positive price rows removed: 0
  impossible OHLC rows removed: 0
  negative volume rows removed: 434
  gaps (kept, not filled): {'unexpected_long': 2}
  spike bars flagged (kept): 51
  gaps (kept, not filled): {'unexpected_long': 2}
  spike bars flagged (kept): 51
```

### 3. Resample Data

Convert to different timeframes (M15, H1, D) with proper timezone handling.

```bash
# Resample to 15-minute bars
python -m data.resample /path/to/your/cleaned.csv --to M15

# Resample to hourly
python -m data.resample /path/to/your/cleaned.csv --to H1

# Resample to daily
python -m data.resample /path/to/your/cleaned.csv --to D
```

## Key Features

### ⏱️ Timezone-Aware Processing

**Auto-detection** - Infers timezone from data patterns
**Named timezones** - Supports all IANA timezone names (e.g., `Europe/Athens`)
**Fixed offsets** - Optional explicit offsets with warnings
**DST handling** - Correctly handles US Daylight Saving Time transitions
**UTC normalization** - All data stored in UTC for consistency

### 🧹 Data Validation

**Duplicate removal** - Removes duplicate timestamps
**NaN handling** - Removes rows with missing OHLC data
**Price validation** - Ensures `high >= open, close, low` and `low <= open, close, high`
**Volume validation** - Removes negative volume entries
**Gap detection** - Identifies breaks and holiday periods
**Spike detection** - Flags extreme price movements

### ⏱️ Resampling

**Complete bars only** - Drops incomplete final bars
**No fake data** - Does not forward-fill gaps
**Timezone-safe** - Maintains proper timezone alignment
**Aggregation** - Custom OHLC aggregation logic

## Testing

Run the test suite to verify data layer integrity:

```bash
python -m pytest tests/test_data_layer.py
```

Tests cover:
- CSV loading with various formats
- Timezone conversion and detection
- Data validation logic
- Gap and spike detection
- Bar resampling
- Edge cases (empty data, DST transitions)
