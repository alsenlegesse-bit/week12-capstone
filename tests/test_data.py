"""Unit tests for data loading module."""

import pytest
from pathlib import Path
import pandas as pd
import tempfile
from src.data.loader import DataLoader, DataConfig

class TestDataLoader:
    """Test suite for DataLoader."""
    
    @pytest.fixture
    def sample_config(self):
        """Fixture providing sample configuration."""
        return DataConfig(
            data_path=Path("test.csv"),
            target_column="target"
        )
    
    def test_initialization(self, sample_config):
        """Test that DataLoader initializes correctly."""
        loader = DataLoader(sample_config)
        assert loader.config.target_column == "target"
        assert loader.config.test_size == 0.2
    
    def test_config_default_values(self):
        """Test default values in DataConfig."""
        config = DataConfig(Path("test.csv"))
        assert config.test_size == 0.2
        assert config.random_state == 42
        assert config.sep == ","
    
    def test_load_data_raises_error_for_missing_file(self, sample_config):
        """Test error handling for missing files."""
        loader = DataLoader(sample_config)
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    def test_data_summary_with_valid_data(self):
        """Test data summary generation."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,target\n1,2,0\n3,4,1\n5,6,0")
            temp_path = f.name
        
        config = DataConfig(Path(temp_path), target_column="target")
        loader = DataLoader(config)
        summary = loader.get_data_summary()
        
        assert summary["shape"] == (3, 3)
        assert "target" in summary["columns"]
        
        # Cleanup
        Path(temp_path).unlink()
    
    def test_split_data(self):
        """Test data splitting functionality."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("col1,col2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0")
            temp_path = f.name
        
        config = DataConfig(Path(temp_path), target_column="target", test_size=0.2)
        loader = DataLoader(config)
        X_train, X_test, y_train, y_test = loader.split_data()
        
        assert len(X_train) == 4
        assert len(X_test) == 1
        assert len(y_train) == 4
        assert len(y_test) == 1
        
        Path(temp_path).unlink()
