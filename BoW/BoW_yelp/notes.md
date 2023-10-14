# Notes BoW Yelp


## Techniques to improve

- Remove stop words
- add a value to each word
- vary classifier

- give more weight to capital words
- smoothing techniques like handling unknown words and unique words (rare words)
 
- use linear layer at end 


## Other stuff to do

- store preprocessed data in a csv or json

## Models

### log_reg_01 : Logistic Regression

- ignore all unknown words
- did not remove stop words
- value of all words is equal
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.5284
            precision    recall  f1-score   support

          0       0.69      0.71      0.70      1141
          1       0.47      0.44      0.46      1009
          2       0.43      0.48      0.45      1003
          3       0.46      0.36      0.40       986
          4       0.56      0.64      0.59       861

    accuracy                           0.53      5000
   macro avg       0.52      0.53      0.52      5000
weighted avg       0.52      0.53      0.52      5000
```

### lin_svc_01 : Linear SVC

- ignore all unknown words
- did not remove stop words
- value of all words is equal
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.4634
              precision    recall  f1-score   support

           0       0.64      0.61      0.63      1141
           1       0.38      0.40      0.39      1009
           2       0.36      0.37      0.37      1003
           3       0.40      0.37      0.38       986
           4       0.53      0.56      0.54       861

    accuracy                           0.46      5000
   macro avg       0.46      0.46      0.46      5000
weighted avg       0.46      0.46      0.46      5000
```

### log_reg 02 : Logistic Regression without stop words

- ignore all unknown words
- value of all words is equal
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.5032
              precision    recall  f1-score   support

           0       0.66      0.66      0.66      1141
           1       0.42      0.43      0.42      1009
           2       0.42      0.40      0.41      1003
           3       0.44      0.41      0.43       986
           4       0.55      0.61      0.58       861

    accuracy                           0.50      5000
   macro avg       0.50      0.50      0.50      5000
weighted avg       0.50      0.50      0.50      5000
```

### log_reg_03 : Logistic Regression with freq cutoff = 30

- ignore all unknown words
- value of all words is equal
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.4982
              precision    recall  f1-score   support

           0       0.66      0.65      0.66      1141
           1       0.41      0.42      0.42      1009
           2       0.40      0.40      0.40      1003
           3       0.44      0.42      0.43       986
           4       0.56      0.59      0.58       861

    accuracy                           0.50      5000
   macro avg       0.49      0.50      0.50      5000
weighted avg       0.50      0.50      0.50      5000
```

### log_reg_04 : Logistic Regression with freq cutoff = 10

- ignore all unknown words
- value of all words is equal
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.4962
              precision    recall  f1-score   support

           0       0.66      0.65      0.66      1141
           1       0.40      0.41      0.41      1009
           2       0.40      0.40      0.40      1003
           3       0.44      0.40      0.42       986
           4       0.55      0.61      0.58       861

    accuracy                           0.50      5000
   macro avg       0.49      0.49      0.49      5000
weighted avg       0.50      0.50      0.50      5000
```

### log_reg_05 : Logistic Regression exclamation mark and capital words

- ignore all unknown words
- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.5028
              precision    recall  f1-score   support

           0       0.67      0.67      0.67      1141
           1       0.42      0.41      0.42      1009
           2       0.41      0.42      0.41      1003
           3       0.44      0.40      0.42       986
           4       0.56      0.60      0.58       861

    accuracy                           0.50      5000
   macro avg       0.50      0.50      0.50      5000
weighted avg       0.50      0.50      0.50      5000
```

### log_reg_06 : Logistic Regression handling unknown words 100 itr

- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.514
              precision    recall  f1-score   support

           0       0.67      0.69      0.68      1141
           1       0.44      0.45      0.45      1009
           2       0.43      0.41      0.42      1003
           3       0.44      0.40      0.42       986
           4       0.55      0.61      0.58       861

    accuracy                           0.51      5000
   macro avg       0.51      0.51      0.51      5000
weighted avg       0.51      0.51      0.51      5000
```

### log_reg_06 : Logistic Regression handling unknown words 200 itr

- no smoothing techniques are used
- Metrics

```txt
Accuracy: 0.497
              precision    recall  f1-score   support

           0       0.68      0.66      0.67      1141
           1       0.41      0.42      0.41      1009
           2       0.40      0.40      0.40      1003
           3       0.43      0.40      0.42       986
           4       0.56      0.59      0.57       861

    accuracy                           0.50      5000
   macro avg       0.49      0.49      0.49      5000
weighted avg       0.50      0.50      0.50      5000
```

### lin_layer_01: Linear Layer 4 epochs

- no smoothing techniques are used
- Metrics

```txt
Accuracy of the network on the 50000 test inputs: 57.102 %
Confusion Matrix:
[[7948 1381  325   85  261]
 [3030 4229 2061  341  339]
 [ 681 1844 4597 1930  948]
 [ 185  297 1774 3729 4015]
 [ 201   99  335 1317 8048]]
Classification Report:
              precision    recall  f1-score   support

           0       0.66      0.79      0.72     10000
           1       0.54      0.42      0.47     10000
           2       0.51      0.46      0.48     10000
           3       0.50      0.37      0.43     10000
           4       0.59      0.80      0.68     10000

    accuracy                           0.57     50000
   macro avg       0.56      0.57      0.56     50000
weighted avg       0.56      0.57      0.56     50000
```
