
# Automated Detection and Classification of Banana Leaf Diseases Using CNN with Spatial Attention Mechanism

## Project Overview

This project focuses on the automated detection and classification of banana leaf diseases using a Convolutional Neural Network (CNN) integrated with a custom spatial attention mechanism called **KernelAttention**.

Banana is an important agricultural crop that contributes to food security and provides income to farming communities. However, several fungal diseases affect banana leaves, reducing photosynthetic efficiency, damaging plant tissue, and potentially lowering crop yields.

Traditional disease identification depends on visual inspection by agricultural specialists. Such approaches can be time-consuming, expensive, and difficult to implement across large plantations.

To address this problem, this research develops a deep learning model capable of classifying banana leaf images into four categories:

1. Healthy
2. Sigatoka
3. Cordana Leaf Spot
4. Pestalotiopsis Leaf Blight

The proposed CNN uses KernelAttention to emphasize disease-related spatial features and improve the learned representation of leaf symptoms.

The model achieved a **97.50% overall test accuracy** on 760 test images.

---

## Key Highlights

| Feature | Description |
|---|---|
| Domain | Artificial Intelligence in Agriculture |
| Approach | Deep Learning |
| Architecture | CNN with KernelAttention |
| Dataset | BananaLSD |
| Original Images | 937 |
| Augmented Images | 1,600 |
| Training Samples | 1,351 |
| Validation Samples | 120 |
| Testing Samples | 760 |
| Input Resolution | 224 × 224 × 3 |
| Number of Classes | 4 |
| Training Epochs | 40 |
| Test Accuracy | 97.50% |
| Explainability Technique | Grad-CAM |
| Evaluation Metrics | Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC |

---

## Problem Statement

Banana leaf diseases can be difficult to identify accurately because several infections produce similar symptoms.

Common disease indicators include:

- Brown and black spots
- Yellowing of leaf tissue
- Irregular lesions
- Necrotic patches
- Changes in leaf texture
- Drying and deterioration of infected regions

Manual diagnosis presents several challenges, including dependence on trained specialists, delays in identifying diseases, and limited accessibility in rural agricultural areas.

A computer vision-based classification system can assist disease screening by learning patterns from labeled banana leaf images.

This project investigates whether integrating a spatial attention mechanism into a CNN can support accurate classification of multiple banana leaf conditions.

---

## Research Objectives

The objectives of this project are:

1. Develop a CNN-based banana leaf disease classification model.
2. Classify four banana leaf conditions using digital images.
3. Design and integrate the custom KernelAttention mechanism.
4. Use hierarchical spatial attention to learn discriminative image features.
5. Address dataset imbalance through augmentation.
6. Evaluate the model using standard classification metrics.
7. Analyze classification errors using a confusion matrix.
8. Evaluate class discrimination using ROC-AUC.
9. Apply Grad-CAM for model interpretability.
10. Investigate the potential of attention-based deep learning for agricultural disease identification.

---

## Disease Classification Categories

### 1. Healthy

Healthy banana leaves generally have green foliage without visible disease-related lesions.

The model learns to distinguish normal leaf appearance from patterns associated with fungal infections.

### 2. Sigatoka

Sigatoka is a fungal banana leaf disease associated with dark spots and streaks.

Common symptoms include:

- Dark brown or black spots
- Elongated streaks
- Yellowing of affected tissue
- Enlarging lesions
- Drying and necrosis

Advanced infections can significantly affect leaf health and photosynthetic activity.

### 3. Cordana Leaf Spot

Cordana Leaf Spot is associated with the fungus *Cordana musae*.

Common symptoms include:

- Circular or irregular spots
- Water-soaked lesions
- Brown discoloration
- Drying of infected areas
- Premature leaf deterioration

### 4. Pestalotiopsis Leaf Blight

Pestalotiopsis Leaf Blight is associated with fungi belonging to the genus *Pestalotiopsis*.

Common symptoms include:

- Irregular brown patches
- Water-soaked regions
- Expanding lesions
- Necrotic leaf tissue
- Merging diseased areas

Some symptoms overlap with other fungal diseases, making classification more challenging.

---

## Dataset

### BananaLSD Dataset

This research uses the BananaLSD dataset, containing banana leaf photographs collected under natural agricultural field conditions.

The dataset was collected in the Gazipur region of Bangladesh using smartphone cameras.

It contains variations in:

- Lighting
- Leaf orientation
- Camera angle
- Background
- Image composition
- Disease appearance

These characteristics provide realistic agricultural images for model development and evaluation.

### Original Dataset Composition

| Disease Category | Original Images | Augmented Images |
|---|---:|---:|
| Pestalotiopsis Leaf Blight | 173 | 400 |
| Sigatoka | 473 | 400 |
| Cordana Leaf Spot | 162 | 400 |
| Healthy | 129 | 400 |
| **Total** | **937** | **1,600** |

The original dataset contains an unequal distribution of images across the four categories.

Sigatoka has the largest number of original images, while Healthy has the smallest.

### Final Experimental Dataset

| Dataset Split | Images |
|---|---:|
| Training | 1,351 |
| Validation | 120 |
| Testing | 760 |
| **Total** | **2,231** |

The test dataset contains 190 images per class.

---

## Data Preprocessing

All input images are standardized to a resolution of:

`224 × 224 × 3`

Where:

- 224 represents image height.
- 224 represents image width.
- 3 represents the RGB color channels.

Standardized image dimensions allow the CNN to process samples consistently.

---

## Data Augmentation

Data augmentation was used to increase the diversity of training examples and reduce the influence of class imbalance.

The augmentation techniques described in the research include:

| Technique | Purpose |
|---|---|
| Rotation | Handles variation in leaf orientation |
| Horizontal Flipping | Adds orientation diversity |
| Width Shifting | Simulates horizontal displacement |
| Height Shifting | Simulates vertical displacement |
| Shearing | Represents geometric distortion |
| Zooming | Handles different camera distances |
| Fill Mode | Fills empty regions after transformations |

Augmentation helps expose the model to variations that may appear in agricultural environments.

---

## Proposed Methodology

The proposed methodology consists of the following stages:

1. Dataset collection and preparation
2. Image preprocessing
3. Data augmentation
4. Training, validation, and test dataset preparation
5. CNN-based feature extraction
6. KernelAttention-based spatial feature weighting
7. Disease classification using Softmax
8. Performance evaluation
9. Explainability analysis using Grad-CAM

The architecture progressively learns features from the banana leaf images and uses attention blocks to emphasize spatially relevant feature representations.

---

## Model Architecture

The proposed CNN architecture includes an initial feature extraction block, three KernelAttention blocks, and a classification head.

### Architecture Components

| Layer | Configuration |
|---|---|
| Input | 224 × 224 × 3 |
| Initial Convolution | 32 filters, 3 × 3 |
| Normalization | Batch Normalization |
| Activation | ReLU |
| Pooling | MaxPooling2D, 2 × 2 |
| KernelAttention Block 1 | 64 filters |
| KernelAttention Block 2 | 128 filters |
| KernelAttention Block 3 | 256 filters |
| Feature Aggregation | Global Average Pooling |
| Dense Layer | 64 units, ReLU |
| Dropout | 0.5 |
| Output Layer | 4 units, Softmax |

### Initial Convolution Layer

The initial convolution layer uses 32 filters of size 3 × 3 to detect basic image features.

These features include edges, textures, and visual patterns associated with leaf structure.

### Batch Normalization

Batch normalization standardizes intermediate activations and helps stabilize training.

### ReLU Activation

The Rectified Linear Unit introduces nonlinearity.

Mathematical expression:

`ReLU(x) = max(0, x)`

### Max Pooling

Max pooling reduces the spatial dimensions of feature maps.

The model uses a 2 × 2 pooling window.

### KernelAttention Blocks

Three KernelAttention blocks contain 64, 128, and 256 filters, respectively.

These blocks support hierarchical feature learning while applying spatial attention.

### Global Average Pooling

Global average pooling converts spatial feature maps into a compact feature representation.

### Fully Connected Classification Layer

A dense layer containing 64 neurons processes the extracted features.

A dropout rate of 0.5 is used before the final prediction layer.

### Softmax Output

The final dense layer contains four output units corresponding to the classification categories.

The Softmax function produces normalized probabilities for each category.

---

## KernelAttention Spatial Attention Mechanism

### Main Research Contribution

The proposed architecture introduces a custom spatial attention mechanism called **KernelAttention**.

A conventional CNN extracts visual features through convolutional operations, but the resulting feature maps may contain information from both disease-relevant and less relevant regions.

KernelAttention learns attention weights that can emphasize spatial locations containing useful classification features.

### Query, Key, and Value Representations

The mechanism generates three convolution-based representations:

- Query (Q)
- Key (K)
- Value (V)

These representations are processed through convolution, batch normalization, and ReLU activation.

### Step 1: Input Projection

If input and output channel dimensions differ, a 1 × 1 convolution projects the input to the required dimensions.

Otherwise, the original input is preserved.

### Step 2: Query Generation

`Q = ReLU(BN(Conv3×3_Q(X)))`

### Step 3: Key Generation

`K = ReLU(BN(Conv3×3_K(X)))`

### Step 4: Value Generation

`V = ReLU(BN(Conv3×3_V(X)))`

### Step 5: Attention Map Generation

The query and key representations are combined:

`A_intermediate = Q + K`

A 1 × 1 convolution followed by sigmoid activation produces the attention map:

`A = sigmoid(Conv1×1(A_intermediate))`

The sigmoid function produces weights between 0 and 1.

### Step 6: Applying Attention

The value representation is multiplied element-wise by the attention map:

`C = V ⊙ A`

Here, `⊙` denotes element-wise multiplication.

### Step 7: Residual Connection

The attention-weighted output is combined with the input projection:

`Y = C + X_projected`

The residual connection preserves information from the original feature representation while incorporating attention-weighted features.

### Advantages of KernelAttention

- Applies spatial attention within the CNN.
- Supports hierarchical learning through multiple attention blocks.
- Combines attention-weighted features with residual information.
- Allows the model to assign different weights to spatial feature locations.
- Helps investigate disease-related feature learning.

---

## Model Training

The model was trained and evaluated over **40 epochs**.

The research examines training, validation, and test accuracy and loss values.

### Reported Accuracy

| Metric | Result |
|---|---:|
| Validation Accuracy | 99.17% |
| Test Accuracy | 97.50% |
| Training Epochs | 40 |

The research reports increasing accuracy during training and generally decreasing loss curves.

These results describe the observed performance on the reported dataset splits.

---

## Results and Performance

### Overall Test Accuracy

**97.50%**

The model achieved 97.50% accuracy on the 760-image test set.

### Per-Class Performance

| Class | Accuracy | Precision | Recall | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| Cordana | 99.34% | 99.47% | 97.89% | 99.82% | 98.67% |
| Healthy | 98.68% | 98.91% | 95.79% | 99.65% | 97.33% |
| Pestalotiopsis | 97.76% | 93.03% | 98.42% | 97.54% | 95.65% |
| Sigatoka | 99.21% | 98.94% | 97.89% | 99.65% | 98.41% |

### Cordana Performance

Cordana achieved:

- Accuracy: 99.34%
- Precision: 99.47%
- Recall: 97.89%
- Specificity: 99.82%
- F1-score: 98.67%

### Healthy Performance

Healthy leaf classification achieved:

- Accuracy: 98.68%
- Precision: 98.91%
- Recall: 95.79%
- Specificity: 99.65%
- F1-score: 97.33%

### Pestalotiopsis Performance

Pestalotiopsis achieved:

- Accuracy: 97.76%
- Precision: 93.03%
- Recall: 98.42%
- Specificity: 97.54%
- F1-score: 95.65%

The comparatively lower precision indicates that some samples from other categories were incorrectly predicted as Pestalotiopsis.

### Sigatoka Performance

Sigatoka achieved:

- Accuracy: 99.21%
- Precision: 98.94%
- Recall: 97.89%
- Specificity: 99.65%
- F1-score: 98.41%

---

## Confusion Matrix Analysis

The confusion matrix shows the distribution of correct and incorrect classifications across the four categories.

| Actual / Predicted | Cordana | Healthy | Pestalotiopsis | Sigatoka |
|---|---:|---:|---:|---:|
| Cordana | 186 | 0 | 4 | 0 |
| Healthy | 0 | 182 | 7 | 1 |
| Pestalotiopsis | 0 | 2 | 187 | 1 |
| Sigatoka | 1 | 0 | 3 | 186 |

### Observations

- Cordana: 186 of 190 samples correctly classified.
- Healthy: 182 of 190 samples correctly classified.
- Pestalotiopsis: 187 of 190 samples correctly classified.
- Sigatoka: 186 of 190 samples correctly classified.

The most frequent misclassification involves Healthy images predicted as Pestalotiopsis.

The confusion matrix contains 741 correct predictions out of 760 test samples.

---

## ROC-AUC Evaluation

Receiver Operating Characteristic (ROC) analysis evaluates the classifier's ability to distinguish between classes across decision thresholds.

The Area Under the Curve (AUC) summarizes this discriminatory ability.

### Reported AUC Results

| Metric | AUC |
|---|---:|
| Micro-average | 0.999 |
| Macro-average | 0.999 |
| Cordana | 0.999 |
| Healthy | 0.999 |
| Pestalotiopsis | 0.998 |
| Sigatoka | 1.000 |

The reported values indicate strong class discrimination on the test dataset.

The Sigatoka AUC of 1.000 indicates perfect ranking discrimination in the reported ROC evaluation, not zero classification errors at the chosen threshold.

---

## Grad-CAM Explainability

Gradient-weighted Class Activation Mapping (Grad-CAM) is used to investigate which image regions influence the model's predictions.

Grad-CAM uses gradient information from convolutional feature maps to identify regions contributing to a particular classification.

### Purpose

Grad-CAM helps researchers assess whether the model focuses on disease-related visual characteristics such as:

- Dark spots
- Lesions
- Necrotic patches
- Discoloration
- Leaf texture
- Disease-associated streaks

### Benefits

1. Improves visibility into model behavior.
2. Helps investigate misclassified examples.
3. Supports examination of influential spatial regions.
4. Helps identify possible attention to irrelevant backgrounds.
5. Provides useful information for future model validation.

Grad-CAM is an interpretability technique and does not independently establish that all influential regions are biologically correct disease indicators.

---

## Comparison With Existing Research

The research paper reviews several previous approaches to banana leaf and plant disease classification.

| Method | Reported Accuracy |
|---|---:|
| BananaSqueezeNet | 96.25% |
| Eight-Layer DCNN | 98.92% |
| DenseNet-201 | 98.12% |
| VGG16 | 97.81% |
| EfficientNet-B0 | 87.81% |
| ResNet50 | 94.00% |
| VGG19 | 94.00% |
| TGVFCMS + CNN | 93.45% |
| ANN + SIFT | 95.00% |
| Proposed CNN + KernelAttention | **97.50%** |

These values come from different published studies.

Because datasets, class definitions, and evaluation methods may differ, the results should not be interpreted as a direct controlled benchmark.

---

## Key Research Contributions

### Custom KernelAttention

Introduces a convolution-based attention mechanism using query, key, and value representations.

### Hierarchical Attention

Incorporates three attention blocks at different levels of the CNN.

### Four-Class Classification

Classifies Healthy, Sigatoka, Cordana Leaf Spot, and Pestalotiopsis Leaf Blight images.

### Data Augmentation

Uses augmentation to improve training sample diversity and address class imbalance.

### Performance Evaluation

Analyzes the model using several classification metrics, confusion matrix results, and ROC-AUC scores.

### Explainability

Uses Grad-CAM to inspect influential image regions.

---

## Limitations

The research has several limitations:

1. The original dataset contains only 937 images.
2. Image collection is concentrated in a limited geographic area.
3. Certain fungal diseases share similar visual symptoms.
4. The classifier only supports four categories.
5. Independent evaluation under additional field conditions is needed.
6. Differences in lighting and camera quality could affect predictions.
7. Controlled ablation experiments are needed to isolate the contribution of KernelAttention.

---

## Future Scope

Possible future improvements include:

- Developing a web application for banana leaf image classification.
- Creating a mobile application for field use.
- Expanding the dataset with additional disease categories.
- Improving performance on difficult and ambiguous examples.
- Introducing disease severity estimation.
- Optimizing the architecture for mobile and edge devices.
- Evaluating the model on independent agricultural datasets.
- Comparing KernelAttention with other attention mechanisms.
- Performing controlled ablation studies.
- Conducting real-world field validation.

These are potential extensions and are not claimed as completed implementations.

---

## Technical Concepts

The research involves:

- Artificial Intelligence
- Deep Learning
- Convolutional Neural Networks
- Computer Vision
- Spatial Attention
- Residual Connections
- Batch Normalization
- Max Pooling
- Global Average Pooling
- Softmax Classification
- Data Augmentation
- ROC-AUC
- Confusion Matrix
- Grad-CAM
- Explainable AI

---

## Repository Structure

The repository contains the project documentation and research paper.

```text
Banana-Leaf-Disease-Classification/
|
|-- README.md
|
|-- project.pdf
```

Source code, trained weights, and dataset files are not listed as available in this repository.

---

## Research Paper

**Title:** Automated Detection and Classification of Banana Leaf Diseases Using Convolutional Neural Network With Spatial Attention Mechanism

The paper describes the dataset, methodology, architecture, mathematical formulation, experimental evaluation, and research findings.

**Full Research Paper:** [project.pdf](project.pdf)

---

## Research Authors

**Hitesh Shivkumar**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Hriday Pathak**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Koushik Daggupati**  
School of Computer Science and Engineering  
Vellore Institute of Technology

**Dr. Reenu Rani**  
School of Advanced Sciences  
Vellore Institute of Technology

---

## Potential Applications

The proposed approach may support future development of:

- Automated agricultural disease screening systems
- Banana plantation health monitoring tools
- AI-assisted plant disease research
- Computer vision-based agricultural applications
- Explainable AI decision-support systems

The reported model is a research prototype and requires additional field validation before operational agricultural deployment.

---

## Conclusion

This project investigates automated banana leaf disease classification using a CNN integrated with the custom KernelAttention spatial attention mechanism.

The model distinguishes four banana leaf categories and achieved a reported test accuracy of **97.50%**.

The evaluation includes class-wise performance measurements, confusion matrix analysis, ROC-AUC scores, and Grad-CAM interpretability.

The results provide a research basis for further investigation into spatial attention and computer vision-based crop disease classification.

---

## Contact

**Koushik Daggupati**

GitHub: https://github.com/DGSKOUSHIK

Repository: https://github.com/DGSKOUSHIK/Banana-Leaf-Disease-Classification
