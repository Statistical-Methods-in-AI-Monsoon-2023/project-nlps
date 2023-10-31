# Notes LSTM Amazon Polarity

## Baseline Pytorch

- 2 epochs

```txt
Test Accuracy: 93.49%
              precision    recall  f1-score   support

           0       0.83      0.95      0.88        20
           1       0.95      0.84      0.89        25

    accuracy                           0.89        45
   macro avg       0.89      0.90      0.89        45
weighted avg       0.90      0.89      0.89        45

Confusion matrix:
[[19  1]
 [ 4 21]]
```

## Baseline Keras

- 2 epochs

```txt
Test Accuracy: 93.58487533401643
              precision    recall  f1-score   support

           0       0.93      0.94      0.94     24530
           1       0.94      0.93      0.94     25243

    accuracy                           0.94     49773
   macro avg       0.94      0.94      0.94     49773
weighted avg       0.94      0.94      0.94     49773

Confusion matrix:
[[23164  1366]
 [ 1827 23416]]
```
