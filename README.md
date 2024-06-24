# Classification of histology images 

This Python project involves the use of an image classification model to identify different histopathological patterns in testicular tissue. 

## Input
TIF files (Immunohistochemistry images, magnification: X10, X20, X40)

## Output
df, CSV file

| File                           | Predicted Phenotype |
|--------------------------------|---------------------|
| Hypospermatogenesis_1-1.jpg    | ['HYPO']            |
| Hypospermatogenesis_1-2.jpg    | ['HYPO']            |
| Normal_Spermatogenesis_1-5.jpg | ['NORSP']           |
| Normal_Spermatogenesis_2-1.jpg | ['NORSP']           |
| SCO_1-4.jpg                    | ['SCO']             |
| SCO_2-1.jpg                    | ['SCO']             |
| SCO_3-1.jpg                    | ['SCO']             |
| SCO_4-3.jpg                    | ['SCO']             |
| Tubular_Shadows_3-4.jpg        | ['TUBS']            |
| Tubular_Shadows_3-5.jpg        | ['TUBS']            |

## Data Preparation
Immunohistochemistry images are converted from TIF to JPG and moved to target directories 

## Feature extraction package: img2vec_pytorch
https://github.com/christiansafka/img2vec

## Classifier: RandomForestClassifier
https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

```python
pip install scikit-learn
from sklearn.ensemble import RandomForestClassifier
```
