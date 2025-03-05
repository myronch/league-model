# League Model Data Collection

A LightGBM based system for predicting League of Legends player performance in professional matches. The model combines historical player statistics, team performance data, and recent match history to generate predictions for kills, assists, and deaths.

Key features:
- Player performance prediction using Random Forest regression
- Automated data collection from multiple sources
- Team-vs-team match outcome analysis
- Win probability adjustments
- Fantasy point calculations

## Usage

Run scripts following the example provided in train.ipynb

## Project Structure

- `train.py`: Main model implementation
- `scraper.py`: Web scraping utilities
- `feature_engineering.py`: Feature engineering 
- `train.ipynb`: Running scripts

## To Do:
- [ ] Add match odds as input to KDA predictor
- [ ] Add quartile prediction for confidence interval
- [ ] Automatically scrape Prizepicks for lines and determine what bets to make
