---

# Bowler Recommendation System

This repository contains the main codebase for a machine learning system that recommends the optimal bowler type based on match conditions and historical performance data.

---

## Contents

- Transformer model files (`.pth`) — included in this repo
- Random Forest model files (`.pkl`) — hosted on Google Drive

---

## How to Run (Google Colab)

### Setup

1. Download the following files:

   - `bowler_recommendation_model.py`
   - `final.csv`

2. Run the commands below as needed.

---

## Training the Model

```bash
!python bowler_recommendation_model.py train \
    --data_file final.csv \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 0.0005 \
    --patience 10 \
    --d_model 128 \
    --num_layers 3
```

### Training Output

```
Training model...
Loading data from final.csv...
Bowler types found: ['Right arm Medium' 'Left arm Medium' 'Right arm Spin' 'Left arm Fast'
 'Right arm Fast' 'Left arm Spin']
Analyzing historical performance by bowler type...
Performance analysis completed!
Encoding categorical features...
Target classes: ['Left arm Fast' 'Left arm Medium' 'Left arm Spin' 'Right arm Fast'
 'Right arm Medium' 'Right arm Spin']
Feature dimension: 77
Training set size: 25280
Validation set size: 5417
Test set size: 5418
Starting training...
Epoch 1/50, Train Loss: 1.7237, Val Loss: 1.6502, Val Accuracy: 0.3482
Validation loss decreased (inf --> 1.650231). Saving model...
Epoch 2/50, Train Loss: 1.6798, Val Loss: 1.6500, Val Accuracy: 0.3229
Validation loss decreased (1.650231 --> 1.650004). Saving model...
.
.
.
EarlyStopping counter: 8 out of 10
Epoch 24/50, Train Loss: 1.5762, Val Loss: 1.6381, Val Accuracy: 0.3489
EarlyStopping counter: 9 out of 10
Epoch 25/50, Train Loss: 1.5737, Val Loss: 1.6386, Val Accuracy: 0.3472
EarlyStopping counter: 10 out of 10
Early stopping triggered

Model saved to outputs/experiment_20250507_053632/final_model.pth
Training completed. Model saved to outputs/experiment_20250507_053632/final_model.pth
Experiment directory: outputs/experiment_20250507_053632
```
## Testing Model
```bash
!python bowler_recommendation_model.py test \
    --data_file final.csv \
    --model_path outputs/experiment_20250507_test/final_model.pth \
    --batch_size 32
```
### Test Output
```
Test Results:
Accuracy: 0.3431
                  precision    recall  f1-score   support

   Left arm Fast       0.14      0.16      0.15       347
 Left arm Medium       0.16      0.01      0.02       365
   Left arm Spin       0.23      0.10      0.14       618
  Right arm Fast       0.38      0.48      0.42      1367
Right arm Medium       0.35      0.28      0.31      1413
  Right arm Spin       0.37      0.53      0.43      1308

        accuracy                           0.34      5418
       macro avg       0.27      0.26      0.25      5418
    weighted avg       0.32      0.34      0.32      5418
```

---

## Make Predictions

```bash
!python bowler_recommendation_model.py predict \
    --model_path outputs/experiment_20250507_053632/final_model.pth \
    --predict_from_input \
    --percentage \
    --temperature 32.5 \
    --humidity 70.0 \
    --precipitation 2.0 \
    --wind_speed 15.0 \
    --cloud_cover 40.0 \
    --soil_type black \
    --venue "Wankhede Stadium" \
    --overnumber 18
```

### Prediction Output

```
Making predictions...
Model loaded from outputs/experiment_20250508_124823/final_model.pth

Predicting using user-provided values:

Input features:
temperature: 32.5
humidity: 70.0
precipitation: 2.0
wind_speed: 15.0
cloud_cover: 40.0
soil_type: black
venue: Wankhede Stadium
overnumber: 1

Processed features for prediction:
temperature: 32.5
humidity: 70.0
precipitation: 2.0
wind_speed: 15.0
cloud_cover: 40.0
venue_fast_economy: 8.323369454099366
venue_spin_economy: 7.758156522763645
venue_medium_economy: 8.899484621778885
soil_fast_economy: 8.222021052631579
soil_spin_economy: 7.70934550688516
soil_medium_economy: 8.550019173984737
phase_fast_economy: 7.546130256849127
phase_spin_economy: 7.496655641228944
phase_medium_economy: 7.858845878895391
temp_fast_economy: 8.244418685955786
temp_spin_economy: 7.62743765735425
temp_medium_economy: 8.361192364665953
humidity_fast_economy: 8.247519436069869
humidity_spin_economy: 7.568865462484121
humidity_medium_economy: 8.417103864000659
wind_fast_economy: 8.264852382897221
wind_spin_economy: 7.570899951568606
wind_medium_economy: 8.526396865769609
soil_black: 1
venue_Wankhede Stadium: 1
phase_powerplay: 1

Prediction Result:
Recommended Bowler Type: Right arm Fast
Expected Runs Conceded (per over): 8.13

Likelihood of Each Bowler Type:
Left arm Fast: 17.82%
Left arm Medium: 13.76%
Left arm Spin: 5.78%
Right arm Fast: 33.05%
Right arm Medium: 19.49%
Right arm Spin: 10.11%
```

---
