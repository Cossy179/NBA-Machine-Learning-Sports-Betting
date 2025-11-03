"""
Time Series Validation Utilities for NBA ML Models
Implements walk-forward validation and time-based cross-validation to prevent look-ahead bias.

Research shows time-based CV is critical for realistic performance estimates in temporal data.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Tuple, Callable, Dict, Optional, Generator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def create_time_based_splits(
    dates: pd.Series,
    n_splits: int = 5,
    gap_days: int = 0,
    min_train_size: Optional[int] = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create time-based train/validation splits respecting temporal order.
    
    Parameters:
    -----------
    dates : pd.Series
        Datetime series for each sample
    n_splits : int, default=5
        Number of splits to create
    gap_days : int, default=0
        Gap between train and validation (prevents data leakage)
    min_train_size : int, optional
        Minimum size of training set
        
    Returns:
    --------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) tuples
    """
    dates = pd.to_datetime(dates)
    
    # Sort by date
    sorted_indices = dates.argsort().values
    sorted_dates = dates.iloc[sorted_indices]
    
    # Use sklearn's TimeSeriesSplit
    tscv = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap_days,
        max_train_size=None  # Use all available training data
    )
    
    splits = []
    for train_idx, val_idx in tscv.split(sorted_indices):
        # Map back to original indices
        train_original = sorted_indices[train_idx]
        val_original = sorted_indices[val_idx]
        
        # Ensure minimum training size
        if min_train_size and len(train_original) < min_train_size:
            continue
            
        splits.append((train_original, val_original))
    
    return splits


def create_season_based_splits(
    dates: pd.Series,
    test_seasons: List[int] = [2023, 2024],
    validation_season: Optional[int] = None,
    season_start_month: int = 10
) -> Dict[str, np.ndarray]:
    """
    Create splits based on NBA seasons.
    
    NBA seasons typically run from October to June.
    
    Parameters:
    -----------
    dates : pd.Series
        Datetime series for each sample
    test_seasons : List[int]
        Seasons to use for testing (e.g., [2023, 2024] for 2023-24, 2024-25)
    validation_season : int, optional
        Season to use for validation. If None, uses last season before test.
    season_start_month : int, default=10
        Month when NBA season starts (October = 10)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with 'train', 'val', and 'test' index arrays
    """
    dates = pd.to_datetime(dates)
    
    # Assign each date to an NBA season
    def get_season_year(date):
        """NBA season year (e.g., 2023 for 2023-24 season)"""
        if date.month >= season_start_month:
            return date.year
        else:
            return date.year - 1
    
    season_years = dates.apply(get_season_year)
    
    # Determine validation season if not specified
    if validation_season is None:
        all_seasons = sorted(season_years.unique())
        # Use season before first test season
        available_seasons = [s for s in all_seasons if s < min(test_seasons)]
        if available_seasons:
            validation_season = available_seasons[-1]
        else:
            validation_season = min(test_seasons) - 1
    
    # Create splits
    test_mask = season_years.isin(test_seasons)
    val_mask = season_years == validation_season
    train_mask = season_years < validation_season
    
    splits = {
        'train': np.where(train_mask)[0],
        'val': np.where(val_mask)[0],
        'test': np.where(test_mask)[0]
    }
    
    # Print split info
    print(f"\n{'='*60}")
    print("SEASON-BASED DATA SPLITS")
    print(f"{'='*60}")
    print(f"Training:   {len(splits['train']):5d} samples (seasons < {validation_season})")
    print(f"Validation: {len(splits['val']):5d} samples (season {validation_season})")
    print(f"Testing:    {len(splits['test']):5d} samples (seasons {test_seasons})")
    print(f"{'='*60}\n")
    
    return splits


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    model_fn: Callable,
    test_seasons: List[int] = [2020, 2021, 2022, 2023, 2024],
    min_train_seasons: int = 5,
    temporal_weights: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Perform walk-forward validation by season.
    
    Train on all data before each test season, test on that season.
    This simulates realistic deployment where you retrain periodically.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : pd.Series
        Dates for each sample
    model_fn : Callable
        Function that trains and returns a model
        Signature: model_fn(X_train, y_train, sample_weights) -> model
    test_seasons : List[int]
        Seasons to test on
    min_train_seasons : int, default=5
        Minimum number of seasons needed for training
    temporal_weights : np.ndarray, optional
        Sample weights for temporal weighting
    verbose : bool, default=True
        Print progress
        
    Returns:
    --------
    Dict[str, List]
        Dictionary containing results for each test season
    """
    dates = pd.to_datetime(dates)
    
    # Get season for each sample
    def get_season_year(date):
        if date.month >= 10:  # October or later
            return date.year
        else:
            return date.year - 1
    
    season_years = dates.apply(get_season_year).values
    
    results = {
        'test_seasons': [],
        'train_sizes': [],
        'test_sizes': [],
        'predictions': [],
        'true_values': [],
        'test_indices': []
    }
    
    for test_season in test_seasons:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Walk-Forward Validation: Testing Season {test_season}")
            print(f"{'='*60}")
        
        # Training: all data before test season
        train_mask = season_years < test_season
        test_mask = season_years == test_season
        
        # Check if we have enough training data
        unique_train_seasons = len(np.unique(season_years[train_mask]))
        if unique_train_seasons < min_train_seasons:
            if verbose:
                print(f"⚠️  Skipping: only {unique_train_seasons} training seasons "
                      f"(need {min_train_seasons})")
            continue
        
        # Check if we have test data
        if test_mask.sum() == 0:
            if verbose:
                print(f"⚠️  Skipping: no data for season {test_season}")
            continue
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Apply temporal weights if provided
        if temporal_weights is not None:
            weights_train = temporal_weights[train_mask]
        else:
            weights_train = None
        
        if verbose:
            print(f"Training: {len(X_train)} samples ({unique_train_seasons} seasons)")
            print(f"Testing:  {len(X_test)} samples")
        
        # Train model
        try:
            model = model_fn(X_train, y_train, weights_train)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_test)
                if y_pred.ndim > 1:
                    y_pred = y_pred[:, 1]  # Probability of positive class
            else:
                y_pred = model.predict(X_test)
            
            # Store results
            results['test_seasons'].append(test_season)
            results['train_sizes'].append(len(X_train))
            results['test_sizes'].append(len(X_test))
            results['predictions'].append(y_pred)
            results['true_values'].append(y_test)
            results['test_indices'].append(np.where(test_mask)[0])
            
            if verbose:
                print(f"✅ Successfully trained and tested on season {test_season}")
                
        except Exception as e:
            if verbose:
                print(f"❌ Error training/testing on season {test_season}: {e}")
            continue
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Walk-Forward Validation Complete")
        print(f"Successfully tested on {len(results['test_seasons'])} seasons")
        print(f"{'='*60}\n")
    
    return results


def expanding_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    n_splits: int = 5,
    min_train_size: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate expanding window cross-validation splits.
    
    Training window expands over time, test window moves forward.
    Similar to TimeSeriesSplit but with explicit date handling.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : pd.Series
        Dates for each sample
    n_splits : int, default=5
        Number of splits
    min_train_size : int, optional
        Minimum training size
        
    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices) for each split
    """
    dates = pd.to_datetime(dates)
    sorted_idx = dates.argsort().values
    n_samples = len(sorted_idx)
    
    # Determine test window size
    test_size = n_samples // (n_splits + 1)
    
    # Set minimum training size
    if min_train_size is None:
        min_train_size = test_size
    
    for i in range(n_splits):
        # Training: expanding window from start
        train_end = min_train_size + (i * test_size)
        train_idx = sorted_idx[:train_end]
        
        # Test: next window
        test_start = train_end
        test_end = min(train_end + test_size, n_samples)
        test_idx = sorted_idx[test_start:test_end]
        
        if len(test_idx) > 0:
            yield train_idx, test_idx


def rolling_window_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.Series,
    train_window_size: int,
    test_window_size: int,
    step_size: Optional[int] = None
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate rolling window cross-validation splits.
    
    Both training and test windows have fixed size and roll forward.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    dates : pd.Series
        Dates for each sample
    train_window_size : int
        Size of training window
    test_window_size : int
        Size of test window
    step_size : int, optional
        Step size for rolling (default: test_window_size)
        
    Yields:
    -------
    Tuple[np.ndarray, np.ndarray]
        (train_indices, test_indices) for each split
    """
    if step_size is None:
        step_size = test_window_size
    
    dates = pd.to_datetime(dates)
    sorted_idx = dates.argsort().values
    n_samples = len(sorted_idx)
    
    start = 0
    while start + train_window_size + test_window_size <= n_samples:
        train_idx = sorted_idx[start:start + train_window_size]
        test_idx = sorted_idx[
            start + train_window_size:start + train_window_size + test_window_size
        ]
        
        yield train_idx, test_idx
        start += step_size


class TemporalValidator:
    """
    Comprehensive temporal validation class with multiple strategies.
    """
    
    def __init__(self, dates: pd.Series):
        """
        Initialize temporal validator.
        
        Parameters:
        -----------
        dates : pd.Series
            Datetime series for all samples
        """
        self.dates = pd.to_datetime(dates)
        self.sorted_indices = self.dates.argsort().values
        
    def get_train_val_test_split(
        self,
        train_end_date: str,
        val_end_date: str,
        test_end_date: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Split data based on specific dates.
        
        Parameters:
        -----------
        train_end_date : str
            End date for training period (exclusive)
        val_end_date : str
            End date for validation period (exclusive)
        test_end_date : str, optional
            End date for test period (exclusive). If None, uses all remaining data.
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with 'train', 'val', 'test' index arrays
        """
        train_end = pd.to_datetime(train_end_date)
        val_end = pd.to_datetime(val_end_date)
        
        train_mask = self.dates < train_end
        val_mask = (self.dates >= train_end) & (self.dates < val_end)
        
        if test_end_date:
            test_end = pd.to_datetime(test_end_date)
            test_mask = (self.dates >= val_end) & (self.dates < test_end)
        else:
            test_mask = self.dates >= val_end
        
        return {
            'train': np.where(train_mask)[0],
            'val': np.where(val_mask)[0],
            'test': np.where(test_mask)[0]
        }
    
    def get_cross_validation_splits(
        self,
        n_splits: int = 5,
        strategy: str = 'expanding'
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get cross-validation splits using specified strategy.
        
        Parameters:
        -----------
        n_splits : int, default=5
            Number of splits
        strategy : str, default='expanding'
            'expanding' - expanding window (recommended)
            'rolling' - rolling window with fixed size
            
        Returns:
        --------
        List[Tuple[np.ndarray, np.ndarray]]
            List of (train_indices, val_indices)
        """
        if strategy == 'expanding':
            tscv = TimeSeriesSplit(n_splits=n_splits)
            return [(train_idx, val_idx) for train_idx, val_idx 
                    in tscv.split(self.sorted_indices)]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def print_split_summary(self, splits: Dict[str, np.ndarray]):
        """Print summary of data splits."""
        print(f"\n{'='*60}")
        print("DATA SPLIT SUMMARY")
        print(f"{'='*60}")
        
        for split_name, indices in splits.items():
            if len(indices) > 0:
                dates_split = self.dates.iloc[indices]
                print(f"{split_name.upper():12s}: {len(indices):5d} samples  "
                      f"({dates_split.min().date()} to {dates_split.max().date()})")
            else:
                print(f"{split_name.upper():12s}: {len(indices):5d} samples  (empty)")
        
        print(f"{'='*60}\n")


def prevent_data_leakage_check(
    train_dates: pd.Series,
    test_dates: pd.Series,
    gap_days: int = 0
) -> bool:
    """
    Check if there's temporal leakage between train and test sets.
    
    Parameters:
    -----------
    train_dates : pd.Series
        Training set dates
    test_dates : pd.Series
        Test set dates
    gap_days : int, default=0
        Required gap between train and test
        
    Returns:
    --------
    bool
        True if no leakage detected, False otherwise
    """
    train_dates = pd.to_datetime(train_dates)
    test_dates = pd.to_datetime(test_dates)
    
    max_train_date = train_dates.max()
    min_test_date = test_dates.min()
    
    # Check for leakage
    if max_train_date >= min_test_date:
        print(f"⚠️  DATA LEAKAGE DETECTED!")
        print(f"   Max train date: {max_train_date.date()}")
        print(f"   Min test date:  {min_test_date.date()}")
        return False
    
    # Check gap
    gap = (min_test_date - max_train_date).days
    if gap < gap_days:
        print(f"⚠️  INSUFFICIENT GAP!")
        print(f"   Actual gap: {gap} days")
        print(f"   Required:   {gap_days} days")
        return False
    
    print(f"✅ No data leakage detected (gap: {gap} days)")
    return True

