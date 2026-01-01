# Contributing Guidelines

## Welcome! 👋

Thank you for interest in contributing to the ARIMA Forecasting Dashboard. This guide will help you get started.

## Code of Conduct

- Be respectful and professional
- Value all contributions
- Support inclusive environment
- Report issues appropriately

## Getting Started

### 1. Fork the Repository

```bash
git clone https://github.com/yourusername/arima-forecasting-dashboard.git
cd arima-forecasting-dashboard
```

### 2. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Set Up Development Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov black flake8
```

## Development Workflow

### Code Style

Follow PEP 8 conventions:

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Type hints
from typing import List, Dict, Optional
def my_function(data: List[str]) -> Dict[str, int]:
    pass
```

### Documentation

All functions must have docstrings:

```python
def calculate_returns(prices: pd.Series, pct: bool = True) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series (time-indexed)
        pct: Return as percentage (True) or decimal (False)
    
    Returns:
        Series of returns
        
    Raises:
        ValueError: If series is empty
        
    Example:
        >>> prices = pd.Series([100, 110, 105])
        >>> returns = calculate_returns(prices)
    """
    pass
```

### Testing

Write tests for all new features:

```python
def test_calculate_returns_percentage():
    """Test calculating returns as percentages."""
    prices = pd.Series([100, 110])
    returns = calculate_returns(prices, pct=True)
    assert returns.iloc[1] == 10.0

def test_calculate_returns_empty():
    """Test handling of empty series."""
    empty = pd.Series([])
    with pytest.raises(ValueError):
        calculate_returns(empty)
```

Run tests before submitting:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_helpers.py::TestStatisticalCalculations -v
```

## Areas for Contribution

### 1. Code Improvements

- Bug fixes
- Performance optimization
- Code refactoring
- Type hints

### 2. Features

- New ARIMA variants (SARIMA, ARIMAX)
- Additional visualizations
- Alternative forecasting models
- Enhanced diagnostics

### 3. Documentation

- Tutorial improvements
- API documentation
- Examples and case studies
- Educational materials

### 4. Testing

- Increase test coverage
- Integration tests
- Edge case testing
- Performance benchmarks

### 5. Infrastructure

- CI/CD improvements
- Docker configuration
- GitHub Actions
- Deployment guides

## Pull Request Process

### 1. Before Submitting

```bash
# Update from main
git fetch origin
git rebase origin/main

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
flake8 src/ tests/

# Update documentation
# Edit docs/ files if needed
```

### 2. Commit Message Format

```
[TYPE] Brief description

Detailed explanation of changes.

Fixes: #issue_number
```

Types: `[FEAT]`, `[FIX]`, `[DOCS]`, `[TEST]`, `[REFACTOR]`

Example:
```
[FEAT] Add SARIMA model support

Implements seasonal ARIMA for handling seasonal data.
Includes auto-selection of seasonal parameters.

Fixes: #42
```

### 3. Submit Pull Request

- Clear title and description
- Link related issues
- Include test results
- Update documentation
- Add yourself to CONTRIBUTORS

### 4. Code Review

- Address feedback promptly
- Discuss disagreements respectfully
- Keep PR focused
- Respond to all comments

## Project Structure

```
arima-forecasting-dashboard/
├── src/
│   ├── data/              # Data loading and caching
│   ├── models/            # ARIMA models
│   ├── visualization/     # Charts and plots
│   └── utils/             # Helper functions
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks
└── logs/                  # Log files
```

### Adding New Modules

1. Create in appropriate `src/` subfolder
2. Add `__init__.py`
3. Write comprehensive docstrings
4. Add unit tests in `tests/`
5. Document in `docs/`

Example:

```python
# src/models/new_model.py
"""New forecasting model module."""

class NewModel:
    """Forecasting model using new approach."""
    
    def __init__(self, series, **kwargs):
        """Initialize model."""
        pass
    
    def fit(self):
        """Fit model to data."""
        pass
    
    def forecast(self, steps):
        """Generate forecast."""
        pass
```

## Coding Standards

### Naming Conventions

```python
# Functions and variables: snake_case
def calculate_returns(series):
    monthly_returns = series.resample('M').mean()
    return monthly_returns

# Classes: PascalCase
class DataLoader:
    pass

# Constants: UPPER_SNAKE_CASE
MIN_OBSERVATIONS = 50
DEFAULT_TIMEOUT = 30
```

### Type Hints

Use type hints throughout:

```python
from typing import List, Dict, Optional, Tuple

def forecast(
    series: pd.Series,
    steps: int = 30,
    alpha: Optional[float] = 0.05
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Generate forecast and metrics."""
    pass
```

### Error Handling

```python
# Specific exceptions
try:
    data = loader.load_from_csv(filepath)
except FileNotFoundError:
    print("File not found")
except ValueError as e:
    print(f"Invalid data: {e}")

# Custom exceptions
class InsufficientDataError(Exception):
    """Raised when data length < minimum required."""
    pass
```

## Performance Considerations

### Optimization Guidelines

1. **Avoid nested loops**
   ```python
   # Bad
   for i in range(len(series)):
       for j in range(len(series)):
           pass
   
   # Good
   vectorized_operation(series)
   ```

2. **Use pandas/numpy operations**
   ```python
   # Bad
   result = []
   for val in series:
       result.append(val * 2)
   
   # Good
   result = series * 2
   ```

3. **Cache expensive operations**
   ```python
   @st.cache_data
   def load_data():
       return loader.fetch_from_yfinance(...)
   ```

## Documentation Guidelines

### Update Documentation

When adding features:

1. **Docstrings** - In code
2. **API_REFERENCE.md** - Function reference
3. **USAGE.md** - Usage examples
4. **README.md** - Overview changes

### Writing Examples

```python
Example:
    Forecast NIFTY 50 prices:
    
    >>> from src.data.loader import DataLoader
    >>> from src.models.arima import ARIMAModel
    >>> loader = DataLoader()
    >>> df = loader.fetch_from_yfinance("^NSEI")
    >>> model = ARIMAModel(df)
    >>> forecast = model.forecast(steps=30)
```

## Release Process

1. Update version in `src/__init__.py`
2. Update `CHANGELOG.md`
3. Tag release: `git tag v1.0.0`
4. Push to GitHub

## Getting Help

- **Questions:** Open an issue with `[QUESTION]` tag
- **Bugs:** File bug report with reproduction steps
- **Features:** Suggest enhancements with use cases
- **Documentation:** Improve existing docs

## License

By contributing, you agree to license your work under the same license as the project.

---

**Thank you for contributing!** 🙏

Questions? Open an issue or contact the maintainers.

