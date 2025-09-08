# How to Run the Analysis

## Prerequisites
- Python 3.8+ (I used Python 3.9)
- Internet connection for installing packages

## Steps

1. Open terminal and navigate to the project folder
   - `cd [path-to-HIT140_A2-folder]`

2. Set up virtual environment (optional but recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. Install required packages
   ```bash
   pip install -r requirements.txt
   ```

4. Run the analysis script
   ```bash
   python scripts/generate_figures_and_stats.py --data-dir data --fig-dir figures
   ```

## Output
- Generated figures will be saved in `figures/` folder:
  - `risk_rate_by_season.png`
  - `bat_landings_vs_rat_minutes.png`
- Statistical results printed to console:
  - Chi-square test results
  - Logistic regression summary

5. When finished, deactivate virtual environment
   ```bash
   deactivate
   ```

## Troubleshooting
- If you get `ModuleNotFoundError`, make sure virtual environment is activated and packages are installed
- Make sure you're in the correct directory when running the script