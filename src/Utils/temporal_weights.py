"""
Temporal Weighting Utility for NBA ML Models
Provides sample weights to prioritize recent seasons over historical data
"""
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional


def calculate_temporal_weights(
    dates: Union[pd.Series, np.ndarray, list],
    recent_season_start: int = 2021,
    decay_factor: float = 0.7,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate temporal weights for training samples based on their dates.
    
    Recent seasons (>= recent_season_start) receive full weight (1.0).
    Older seasons receive exponentially decaying weights to reduce their influence.
    
    Parameters:
    -----------
    dates : pd.Series, np.ndarray, or list
        Dates for each training sample
    recent_season_start : int, default=2021
        Starting year for "recent" seasons that get full weight (1.0)
    decay_factor : float, default=0.7
        Exponential decay factor for older seasons (0-1)
        Lower values = more aggressive decay
    normalize : bool, default=True
        Whether to normalize weights so they sum to the number of samples
        
    Returns:
    --------
    np.ndarray
        Array of sample weights matching the length of dates
        
    Examples:
    ---------
    >>> dates = pd.Series(['2023-01-01', '2020-01-01', '2015-01-01'])
    >>> weights = calculate_temporal_weights(dates, recent_season_start=2021)
    >>> # 2023 -> 1.0, 2020 -> 0.7, 2015 -> ~0.17
    """
    # Convert to pandas datetime if not already
    if isinstance(dates, list):
        dates = pd.Series(dates)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    # Convert to datetime
    dates = pd.to_datetime(dates)
    
    # Extract year from each date
    years = dates.dt.year.values
    
    # Calculate weights
    weights = np.ones(len(dates), dtype=np.float32)
    
    for i, year in enumerate(years):
        if year >= recent_season_start:
            # Recent seasons get full weight
            weights[i] = 1.0
        else:
            # Older seasons get exponentially decaying weight
            years_ago = recent_season_start - year
            weights[i] = decay_factor ** years_ago
    
    # Normalize weights if requested
    if normalize:
        # Normalize so sum equals number of samples (maintains total training mass)
        weights = weights * (len(weights) / weights.sum())
    
    return weights


def get_season_year(date: Union[str, pd.Timestamp, datetime]) -> int:
    """
    Get NBA season year from a date.
    NBA seasons run from October to June, so we use the starting year.
    
    Parameters:
    -----------
    date : str, pd.Timestamp, or datetime
        Date to get season year for
        
    Returns:
    --------
    int
        NBA season year (e.g., 2023 for 2023-24 season)
    """
    date = pd.to_datetime(date)
    
    # If date is June or earlier, it belongs to the previous season year
    if date.month <= 6:
        return date.year - 1
    else:
        return date.year


def calculate_season_weights(
    dates: Union[pd.Series, np.ndarray, list],
    recent_seasons: int = 4,
    decay_factor: float = 0.7,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate weights based on complete NBA seasons rather than calendar years.
    
    Parameters:
    -----------
    dates : pd.Series, np.ndarray, or list
        Dates for each training sample
    recent_seasons : int, default=4
        Number of most recent complete seasons to give full weight
    decay_factor : float, default=0.7
        Exponential decay factor for older seasons
    normalize : bool, default=True
        Whether to normalize weights
        
    Returns:
    --------
    np.ndarray
        Array of sample weights
    """
    # Convert to pandas datetime
    if isinstance(dates, list):
        dates = pd.Series(dates)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    dates = pd.to_datetime(dates)
    
    # Get season year for each date
    season_years = dates.apply(get_season_year).values
    
    # Find the most recent season
    max_season = season_years.max()
    recent_season_threshold = max_season - recent_seasons + 1
    
    # Calculate weights
    weights = np.ones(len(dates), dtype=np.float32)
    
    for i, season in enumerate(season_years):
        if season >= recent_season_threshold:
            # Recent seasons get full weight
            weights[i] = 1.0
        else:
            # Older seasons get exponentially decaying weight
            seasons_ago = recent_season_threshold - season
            weights[i] = decay_factor ** seasons_ago
    
    # Normalize weights if requested
    if normalize:
        weights = weights * (len(weights) / weights.sum())
    
    return weights


def print_weight_distribution(
    dates: Union[pd.Series, np.ndarray, list],
    weights: np.ndarray
):
    """
    Print summary of weight distribution by year/season for debugging.
    
    Parameters:
    -----------
    dates : pd.Series, np.ndarray, or list
        Dates for each training sample
    weights : np.ndarray
        Calculated weights
    """
    if isinstance(dates, list):
        dates = pd.Series(dates)
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    dates = pd.to_datetime(dates)
    years = dates.dt.year.values
    
    print("\n" + "="*60)
    print("TEMPORAL WEIGHT DISTRIBUTION")
    print("="*60)
    
    # Group by year and show statistics
    unique_years = sorted(set(years))
    for year in unique_years:
        year_mask = years == year
        year_samples = year_mask.sum()
        avg_weight = weights[year_mask].mean()
        total_weight = weights[year_mask].sum()
        
        print(f"Year {year}: {year_samples:4d} samples, "
              f"avg weight: {avg_weight:.3f}, total weight: {total_weight:.1f}")
    
    print("="*60)
    print(f"Total samples: {len(weights)}")
    print(f"Total weight: {weights.sum():.1f}")
    print(f"Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print("="*60 + "\n")

