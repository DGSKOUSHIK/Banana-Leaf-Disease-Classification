
# 🍌 Automated Detection and Classification of Banana Leaf Diseases Using CNN with Spatial Attention Mechanism

<p align="center">
  <img src="https://img.shields.io/badge/Python-Deep%20Learning-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/CNN-Spatial%20Attention-orange?style=for-the-badge" alt="CNN">
  <img src="https://img.shields.io/badge/Test%20Accuracy-97.50%25-brightgreen?style=for-the-badge" alt="Accuracy">
  <img src="https://img.shields.io/badge/Research-Computer%20Vision-purple?style=for-the-badge" alt="Research">
</p>

<p align="center">
  <b>Automated identification of banana leaf diseases using a custom Convolutional Neural Network integrated with a KernelAttention spatial attention mechanism.</b>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-proposed-methodology">Methodology</a> •
  <a href="#-model-architecture">Architecture</a> •
  <a href="#-results-and-performance">Results</a> •
  <a href="#-grad-cam-explainability">Grad-CAM</a>
</p>

---

## 📌 Overview

Banana is an economically important agricultural crop that contributes to global food security and provides income to millions of farmers.

However, banana production is affected by various diseases, particularly fungal infections that damage leaves and reduce photosynthetic activity.

Traditional identification of banana leaf diseases relies on manual inspection by agricultural experts. These methods can be time-consuming, expensive, and difficult to implement on a large scale.

This project presents an automated deep learning-based approach for detecting and classifying banana leaf diseases using a **Convolutional Neural Network (CNN) integrated with a custom spatial attention mechanism called KernelAttention**.

The proposed model is designed to identify four categories:

1. Healthy Banana Leaves
2. Sigatoka Disease
3. Cordana Leaf Spot
4. Pestalotiopsis Leaf Blight

The KernelAttention mechanism helps the network emphasize important regions of leaf images while learning disease-related visual patterns.

The model achieved a **97.50% test classification accuracy** using a dataset of banana leaf images collected under natural agricultural conditions.

### Key Project Highlights

| Feature | Description |
|---|---|
| Project Domain | Artificial Intelligence in Agriculture |
| Primary Technique | Convolutional Neural Network |
| Proposed Mechanism | KernelAttention Spatial Attention |
| Dataset | BananaLSD |
| Original Images | 937 |
| Augmented Images | 1,600 |
| Final Training Images | 1,351 |
| Validation Images | 120 |
| Testing Images | 760 |
| Input Image Resolution | 224 × 224 |
| Classification Categories | 4 |
| Training Epochs | 40 |
| Test Accuracy | **97.50%** |
| Explainability | Grad-CAM |
| Evaluation | Accuracy, Precision, Recall, F1, Specificity, ROC-AUC |

---

## 🎯 Problem Statement

Banana leaf diseases cause significant damage to agricultural production.

Early-stage infections are often difficult to distinguish because different diseases may share similar visual symptoms, including discoloration, dark spots, necrotic regions, and lesions.

Manual disease identification introduces several difficulties:

- Dependence on trained agricultural specialists
- Delays in identifying diseases
- Limited access to experts in rural farming regions
- Difficulty inspecting large plantations
- Similar visual symptoms among different fungal infections
- Variations in lighting, leaf orientation, and image background

An automated image classification model can help identify disease categories from leaf photographs.

The proposed system addresses this problem through deep learning and spatial attention.

---

## 🎯 Research Objectives

The main objectives are:

1. Design an automated banana leaf disease classification system.
2. Develop a CNN model capable of distinguishing four banana leaf conditions.
3. Integrate a custom KernelAttention mechanism into the CNN.
4. Improve feature extraction by emphasizing important spatial regions.
5. Reduce dataset imbalance using image augmentation.
6. Evaluate classification performance using standard machine learning metrics.
7. Analyze classification errors using a confusion matrix.
8. Generate ROC curves and calculate AUC values.
9. Apply Grad-CAM to visualize image regions influencing predictions.
10. Explore the potential of explainable deep learning for agricultural applications.

---

## 🌿 Banana Leaf Disease Categories

The proposed model classifies banana leaf images into four categories.

### 1. Healthy Banana Leaves

Healthy banana leaves generally have green foliage without visible disease-related lesions.

The model must correctly identify healthy leaves to distinguish normal foliage from infected leaves.

**Research Paper Figure:** Figure 1

[📷 View Healthy Banana Leaf Images](project.pdf#page=1)

### 2. Sigatoka Disease

Sigatoka is a fungal disease that affects banana leaves.

It is associated with dark brown or black spots, discoloration, and damage to leaf tissues.

**Common symptoms:**

- Small dark spots
- Brown or black streaks
- Yellowing of leaf tissue
- Expanding lesions
- Drying and necrosis

Severe infections can reduce the photosynthetic capacity of banana plants.

**Research Paper Figure:** Figure 2

[📷 View Sigatoka Disease Images](project.pdf#page=2)

### 3. Cordana Leaf Spot

Cordana Leaf Spot is associated with the fungus *Cordana musae*.

The disease produces visible lesions that may expand and damage leaf tissue.

**Common symptoms:**

- Circular spots
- Water-soaked lesions
- Brown patches
- Drying of infected areas
- Premature leaf damage

**Research Paper Figure:** Figure 3

[📷 View Cordana Disease Images](project.pdf#page=2)

### 4. Pestalotiopsis Leaf Blight

Pestalotiopsis Leaf Blight is associated with fungi from the genus *Pestalotiopsis*.

It commonly produces irregular brown patches and necrotic leaf regions.

**Common symptoms:**

- Irregular brown lesions
- Water-soaked patches
- Expanding diseased areas
- Necrotic tissue
- Merging lesions

**Research Paper Figure:** Figure 4

[📷 View Pestalotiopsis Disease Images](project.pdf#page=2)

---

## 📊 Dataset

### BananaLSD Dataset

The research uses the **BananaLSD dataset**, containing banana leaf photographs captured under natural field conditions.

The dataset was collected in the Gazipur region of Bangladesh.

Images were acquired using smartphone cameras and include variations in:

- Lighting conditions
- Leaf orientation
- Background objects
- Camera angles
- Disease symptoms
- Image appearance

These characteristics make the dataset suitable for evaluating deep learning approaches using agricultural field images.

### Original Dataset Distribution

| Class | Original Images | Augmented Images |
|---|---:|---:|
| Pestalotiopsis Leaf Blight | 173 | 400 |
| Sigatoka | 473 | 400 |
| Cordana Leaf Spot | 162 | 400 |
| Healthy | 129 | 400 |
| **Total** | **937** | **1,600** |

The original dataset is imbalanced, with Sigatoka having the highest number of original images and Healthy having the lowest.

Data augmentation was used to improve representation across the four categories.

### Final Experimental Data Distribution

| Dataset | Number of Images |
|---|---:|
| Training | 1,351 |
| Validation | 120 |
| Testing | 760 |
| **Total** | **2,231** |

The evaluation test dataset contains 190 images from each class.

### Dataset Distribution Visualization

**Research Paper Figure:** Figure 5

[📊 View Dataset Distribution Graph](project.pdf#page=5)

---

## 🔄 Data Preprocessing and Augmentation

Data preprocessing and augmentation were used to prepare images for model training.

### Image Preprocessing

The images were resized to:

**224 × 224 × 3**

Where:

- 224 represents image height.
- 224 represents image width.
- 3 represents RGB color channels.

Standardized image dimensions allow the CNN to process images consistently.

### Data Augmentation Techniques

| Augmentation | Purpose |
|---|---|
| Rotation | Introduces different leaf orientations |
| Horizontal Flipping | Adds orientation diversity |
| Width Shifting | Simulates horizontal movement |
| Height Shifting | Simulates vertical movement |
| Shearing | Simulates geometric distortions |
| Zooming | Represents different camera distances |
| Fill Mode | Handles empty pixels after transformation |

### Why Augmentation?

The original dataset contains unequal numbers of images for different diseases.

Without class balancing, a model may learn to favor majority classes.

Data augmentation expands the available training examples and helps improve generalization.

The research describes using original images for evaluation and augmentation for the training data.

---

## 🧠 Proposed Methodology

The proposed approach follows an image-based deep learning classification process.

### Complete Classification Workflow

```text
                  Banana Leaf Dataset
                           |
                           v
                    Data Collection
                           |
                           v
                     Data Cleaning
                           |
                           v
                    Image Resizing
                      224 × 224
                           |
                           v
                    Data Augmentation
                           |
                           v
                    Dataset Splitting
                           |
              +------------+------------+
              |            |            |
              v            v            v
           Training    Validation     Testing
              |            |            |
              v            |            |
         CNN Training      |            |
              |            |            |
              v            |            |
        Feature Extraction |            |
              |            |            |
              v            |            |
       KernelAttention     |            |
              |            |            |
              v            |            |
       Classification Head |            |
              |            |            |
              v            v            |
           Model Evaluation             |
              |                         |
              +-------------------------+
                           |
                           v
                 Performance Analysis
                           |
                           v
                Confusion Matrix & ROC
                           |
                           v
                  Grad-CAM Analysis
                           |
                           v
                 Disease Classification
```

The system uses CNN feature extraction and KernelAttention blocks to produce representations suitable for classifying banana leaf conditions.

---

## 🏗️ Model Architecture

The proposed CNN architecture integrates a custom spatial attention mechanism at multiple stages.

The network contains:

1. Input layer
2. Initial convolutional layer
3. Batch normalization
4. ReLU activation
5. Max pooling
6. Three KernelAttention blocks
7. Global average pooling
8. Fully connected classification layer
9. Dropout
10. Softmax output layer

### Architecture Visualization

**Research Paper Figure:** Figure 6

[🧠 View Complete CNN Architecture Diagram](project.pdf#page=6)

### Network Architecture

```text
Input Image
224 × 224 × 3
      |
      v
Conv2D
32 filters, 3 × 3
      |
      v
Batch Normalization
      |
      v
ReLU Activation
      |
      v
MaxPooling2D
2 × 2
      |
      v
KernelAttention
64 filters
      |
      v
MaxPooling2D
2 × 2
      |
      v
KernelAttention
128 filters
      |
      v
MaxPooling2D
2 × 2
      |
      v
KernelAttention
256 filters
      |
      v
MaxPooling2D
2 × 2
      |
      v
Global Average Pooling
      |
      v
Dense Layer
64 neurons, ReLU
      |
      v
Dropout
Rate = 0.5
      |
      v
Dense Layer
4 neurons, Softmax
      |
      v
Predicted Disease Class
```

### Initial Convolution Layer

The first convolution layer uses 32 filters with a kernel size of 3 × 3.

This layer learns low-level image features such as:

- Edges
- Shapes
- Textures
- Color-related patterns

### Batch Normalization

Batch normalization helps stabilize intermediate representations during training.

### ReLU Activation

ReLU introduces nonlinearity into the network.

Its mathematical expression is:

`ReLU(x) = max(0, x)`

### Max Pooling

Max pooling reduces the spatial size of the feature maps.

The model uses a pooling window of 2 × 2.

### Global Average Pooling

Global average pooling converts the feature maps into compact numerical representations.

### Dense Classification Layer

A dense layer containing 64 neurons processes the extracted features.

Dropout with a rate of 0.5 is used before the final output layer.

### Softmax Layer

The final dense layer contains four output neurons.

The Softmax activation function converts output scores into probabilities for the four classes.

---

## 🔬 KernelAttention Spatial Attention Mechanism

### Main Research Contribution

The primary architectural contribution is a custom spatial attention layer called **KernelAttention**.

Traditional CNNs extract visual features using convolution operations.

However, disease-related patterns may occupy only small portions of a leaf image.

KernelAttention is designed to place greater emphasis on relevant spatial features while preserving information from the original feature representation.

### KernelAttention Architecture

**Research Paper Figure:** Figure 7

[🔍 View KernelAttention Architecture Diagram](project.pdf#page=7)

### Core Components

KernelAttention uses:

- Query representation (Q)
- Key representation (K)
- Value representation (V)
- Spatial attention map
- Element-wise multiplication
- Residual connection

### Step 1: Input Projection

The input tensor is projected when the input and output channel dimensions differ.

```text
If Input Channels != Output Channels:

    X_projected = Conv1x1(X)

Otherwise:

    X_projected = X
```

This allows the residual connection to combine compatible feature maps.

### Step 2: Query, Key, and Value Generation

Three convolution-based transformations are applied to the input.

**Query:**

`Q = ReLU(BN(Conv3×3_Q(X)))`

**Key:**

`K = ReLU(BN(Conv3×3_K(X)))`

**Value:**

`V = ReLU(BN(Conv3×3_V(X)))`

Here:

- `BN` represents Batch Normalization.
- `ReLU` represents Rectified Linear Unit.
- `Conv3×3` represents a convolution using a 3 × 3 kernel.

### Step 3: Spatial Attention Map

Query and key features are combined:

`A_intermediate = Q + K`

The combined representation is processed using a 1 × 1 convolution and sigmoid activation:

`A = sigmoid(Conv1×1(A_intermediate))`

The sigmoid function generates attention weights between 0 and 1.

### Step 4: Attention-Weighted Features

The value representation is multiplied by the attention map:

`C = V ⊙ A`

Here, `⊙` represents element-wise multiplication.

### Step 5: Residual Connection

The attention-weighted features are combined with the projected input:

`Y = C + X_projected`

The residual connection helps retain original information while incorporating attention-weighted features.

### KernelAttention Workflow

```text
                   Input Feature Map
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
       Query (Q)       Key (K)        Value (V)
           |              |              |
           v              v              |
         Conv2D         Conv2D           |
           |              |              |
           v              v              |
      BatchNorm       BatchNorm          |
           |              |              |
           v              v              |
          ReLU           ReLU             |
           |              |              |
           +-------+------+              |
                   |                     |
                   v                     |
                Addition                 |
                   |                     |
                   v                     |
                Conv1×1                  |
                   |                     |
                   v                     |
                 Sigmoid                  |
                   |                     |
                   v                     |
               Attention Map             |
                   |                     |
                   +----------+----------+
                              |
                              v
                    Element-wise Multiply
                              |
                              v
                     Attention Features
                              |
                              v
                       Residual Add
                              |
                              v
                       Output Features
```

### Benefits of the Mechanism

- Learns spatially weighted feature representations.
- Provides attention at multiple network stages.
- Combines attention output with residual information.
- Supports hierarchical image feature learning.
- Can emphasize regions associated with disease symptoms.

---

## ⚙️ Model Training

The proposed architecture was trained for **40 epochs**.

Training, validation, and test measurements were examined to analyze convergence.

### Training Accuracy

**Research Paper Figure:** Figure 8

[📈 View Training Accuracy Graph](project.pdf#page=8)

The graph shows the development of model accuracy across the training epochs.

The reported final results include:

- Validation accuracy: 99.17%
- Test accuracy: 97.50%

### Training Loss

**Research Paper Figure:** Figure 9

[📉 View Training Loss Graph](project.pdf#page=9)

The loss curves generally decrease during training, indicating a reduction in prediction errors.

The paper reports that the model demonstrates stable learning behavior with limited signs of overfitting in the reported curves.

---

## 🏆 Results and Performance

The proposed model was evaluated using several classification metrics.

### Overall Test Accuracy

**97.50%**

The model was evaluated using a test dataset containing 760 banana leaf images.

### Performance Metrics

| Disease | Accuracy | Precision | Recall | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| Cordana | 99.34% | 99.47% | 97.89% | 99.82% | 98.67% |
| Healthy | 98.68% | 98.91% | 95.79% | 99.65% | 97.33% |
| Pestalotiopsis | 97.76% | 93.03% | 98.42% | 97.54% | 95.65% |
| Sigatoka | 99.21% | 98.94% | 97.89% | 99.65% | 98.41% |

### Cordana Performance

**Research Paper Figure:** Figure 10

[📊 View Cordana Performance Graph](project.pdf#page=9)

The Cordana class achieved 99.34% reported class-wise accuracy and an F1-score of 98.67%.

### Healthy Leaf Performance

**Research Paper Figure:** Figure 11

[📊 View Healthy Leaf Performance Graph](project.pdf#page=10)

The Healthy class achieved 98.68% reported class-wise accuracy and a recall of 95.79%.

### Sigatoka Performance

**Research Paper Figure:** Figure 12

[📊 View Sigatoka Performance Graph](project.pdf#page=10)

Sigatoka achieved 99.21% reported class-wise accuracy and an F1-score of 98.41%.

### Pestalotiopsis Performance

**Research Paper Figure:** Figure 13

[📊 View Pestalotiopsis Performance Graph](project.pdf#page=10)

Pestalotiopsis achieved 97.76% reported class-wise accuracy.

Its precision of 93.03% is lower than that of the other classes.

This indicates that some images belonging to other categories were predicted as Pestalotiopsis.

---

## 🎯 Confusion Matrix

The confusion matrix evaluates how frequently the model correctly and incorrectly classifies images.

**Research Paper Figure:** Figure 14

[🔍 View Confusion Matrix Heatmap](project.pdf#page=11)

### Confusion Matrix Results

| Actual / Predicted | Cordana | Healthy | Pestalotiopsis | Sigatoka |
|---|---:|---:|---:|---:|
| Cordana | 186 | 0 | 4 | 0 |
| Healthy | 0 | 182 | 7 | 1 |
| Pestalotiopsis | 0 | 2 | 187 | 1 |
| Sigatoka | 1 | 0 | 3 | 186 |

### Analysis

**Cordana:** 186 of 190 images were correctly classified, while four were classified as Pestalotiopsis.

**Healthy:** 182 of 190 images were correctly classified. Seven were classified as Pestalotiopsis and one as Sigatoka.

**Pestalotiopsis:** 187 of 190 images were correctly classified. Two were classified as Healthy and one as Sigatoka.

**Sigatoka:** 186 of 190 images were correctly classified. One was classified as Cordana and three as Pestalotiopsis.

### Key Observations

- The majority of samples are correctly classified.
- Pestalotiopsis receives the highest number of false-positive predictions.
- Some Healthy images are confused with Pestalotiopsis.
- Sigatoka and Cordana demonstrate strong classification performance.

The diagonal of the confusion matrix contains 741 correctly classified test samples.

---

## 📈 ROC Curve and AUC Analysis

The Receiver Operating Characteristic (ROC) curve evaluates the ability of a classifier to distinguish between positive and negative examples at different classification thresholds.

The curve plots:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

Area Under the Curve (AUC) summarizes discrimination performance.

### Overall ROC Performance

**Research Paper Figure:** Figure 15

[📈 View Multi-Class ROC Curve](project.pdf#page=11)

| ROC Metric | Reported AUC |
|---|---:|
| Micro-average | 0.999 |
| Macro-average | 0.999 |

### Class-Wise ROC-AUC

| Disease | AUC |
|---|---:|
| Cordana | 0.999 |
| Healthy | 0.999 |
| Pestalotiopsis | 0.998 |
| Sigatoka | 1.000 |

### Cordana ROC

**Research Paper Figure:** Figure 16

[📈 View Cordana ROC Curve](project.pdf#page=12)

AUC: 0.999

### Healthy ROC

**Research Paper Figure:** Figure 17

[📈 View Healthy ROC Curve](project.pdf#page=12)

AUC: 0.999

### Pestalotiopsis ROC

**Research Paper Figure:** Figure 18

[📈 View Pestalotiopsis ROC Curve](project.pdf#page=12)

AUC: 0.998

### Sigatoka ROC

**Research Paper Figure:** Figure 19

[📈 View Sigatoka ROC Curve](project.pdf#page=12)

AUC: 1.000

The reported AUC results indicate strong class discrimination on the evaluated dataset.

An AUC of 1.000 for Sigatoka does not imply that every Sigatoka sample was classified correctly at the selected decision threshold.

---

## 🔥 Grad-CAM Explainability

Grad-CAM stands for **Gradient-weighted Class Activation Mapping**.

It is a visualization technique used to identify image regions that influence CNN predictions.

Grad-CAM produces heatmaps based on the gradients flowing into selected convolutional feature maps.

### Why Grad-CAM?

A CNN may accurately classify an image, but its internal decision process can be difficult to understand.

Grad-CAM helps researchers inspect whether the network is focusing on relevant features.

For banana leaf disease classification, these may include:

- Brown lesions
- Dark spots
- Yellowing
- Necrotic regions
- Disease-related streaks
- Changes in leaf texture

### Class-Wise Grad-CAM Visualizations

#### Cordana

**Research Paper Figure:** Figure 20

[🔥 View Cordana Grad-CAM](project.pdf#page=13)

#### Healthy Leaves

**Research Paper Figure:** Figure 21

[🔥 View Healthy Leaf Grad-CAM](project.pdf#page=13)

#### Sigatoka

**Research Paper Figure:** Figure 22

[🔥 View Sigatoka Grad-CAM](project.pdf#page=13)

The research describes attention patterns associated with vein-oriented disease streaks.

#### Pestalotiopsis

**Research Paper Figure:** Figure 23

[🔥 View Pestalotiopsis Grad-CAM](project.pdf#page=13)

### Importance of Explainability

Grad-CAM helps with:

1. Understanding model predictions.
2. Visualizing influential regions.
3. Investigating classification errors.
4. Detecting possible attention to irrelevant backgrounds.
5. Supporting further model validation.

Grad-CAM helps interpret predictions, but it does not independently prove that a highlighted region is a biologically correct disease marker.

---

## 🆚 Comparison With Existing Research

The research paper reviews multiple approaches to banana leaf and crop disease classification.

| Model / Method | Reported Accuracy |
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

**Note:** These results come from different published studies and may use different datasets, disease classes, and evaluation procedures.

They are included as literature context and are not direct, controlled benchmark results.

---

## 💡 Key Contributions

### 1. Custom KernelAttention Layer

A spatial attention mechanism using convolution-based query, key, and value transformations.

### 2. Hierarchical Attention

Three KernelAttention blocks integrate attention at different network depths.

### 3. Four-Class Classification

The system distinguishes Healthy, Sigatoka, Cordana, and Pestalotiopsis images.

### 4. Data Augmentation

Augmentation is used to improve training data diversity and address class imbalance.

### 5. Detailed Evaluation

The research includes accuracy, precision, recall, specificity, F1-score, confusion matrix, and ROC-AUC evaluation.

### 6. Explainable AI

Grad-CAM provides visual information about the image regions influencing predictions.

---

## ⚠️ Limitations

The research has several limitations that should be considered before real-world deployment.

### Limited Original Dataset

The original dataset contains 937 images.

Additional data could help evaluate model generalization.

### Geographic Dependence

Images were collected from a specific region of Bangladesh.

Performance on banana leaves from other countries has not been established by the reported results.

### Similar Disease Symptoms

Some disease categories share visual features, leading to occasional misclassification.

### Limited Disease Coverage

The model identifies only four categories and does not cover every disease affecting banana plants.

### Environmental Variations

Different lighting conditions, camera resolutions, backgrounds, and leaf orientations could influence performance.

### External Validation

Independent evaluation on new field datasets would help establish reliability under different real-world conditions.

### Attention Mechanism Evaluation

A controlled ablation study would help determine the improvement specifically attributable to KernelAttention.

---

## 🚀 Future Scope

Potential future improvements include:

- Developing a farmer-friendly web interface.
- Building a mobile application for leaf image classification.
- Extending the model to additional diseases.
- Improving prediction reliability using larger datasets.
- Adding disease severity estimation.
- Implementing real-time image classification.
- Optimizing the model for mobile devices.
- Comparing CNN attention mechanisms with transformer-based architectures.
- Conducting independent field evaluations.
- Performing ablation experiments on the KernelAttention blocks.

These are future development possibilities and are not presented as already implemented features.

---

## 📂 Repository Structure

```text
Banana-Leaf-Disease-Classification/
│
├── README.md
│
└── project.pdf
```

### Files

**README.md**

Contains the complete research project overview, architecture explanation, dataset information, methodology, evaluation results, and links to figures.

**project.pdf**

Contains the research paper with its original figures, tables, graphs, mathematical formulations, and experimental findings.

---

## 📄 Research Paper

**Title:** Automated Detection and Classification of Banana Leaf Diseases Using Convolutional Neural Network With Spatial Attention Mechanism

The full research document includes:

- Introduction
- Related Works
- Dataset Description
- Data Preprocessing
- Model Architecture
- KernelAttention Mechanism
- Mathematical Formulation
- Results and Discussion
- Confusion Matrix
- ROC-AUC Analysis
- Grad-CAM Visualization
- References

### 📖 [Read the Complete Research Paper](project.pdf)

### 🖼️ Research Figures

| Figure | Description | Link |
|---|---|---|
| Fig. 1 | Healthy Banana Leaves | [View](project.pdf#page=1) |
| Fig. 2 | Sigatoka Disease | [View](project.pdf#page=2) |
| Fig. 3 | Cordana Disease | [View](project.pdf#page=2) |
| Fig. 4 | Pestalotiopsis Disease | [View](project.pdf#page=2) |
| Fig. 5 | Dataset Distribution | [View](project.pdf#page=5) |
| Fig. 6 | CNN Architecture | [View](project.pdf#page=6) |
| Fig. 7 | KernelAttention Architecture | [View](project.pdf#page=7) |
| Fig. 8 | Model Accuracy | [View](project.pdf#page=8) |
| Fig. 9 | Model Loss | [View](project.pdf#page=9) |
| Fig. 10 | Cordana Performance | [View](project.pdf#page=9) |
| Fig. 11 | Healthy Performance | [View](project.pdf#page=10) |
| Fig. 12 | Sigatoka Performance | [View](project.pdf#page=10) |
| Fig. 13 | Pestalotiopsis Performance | [View](project.pdf#page=10) |
| Fig. 14 | Confusion Matrix | [View](project.pdf#page=11) |
| Fig. 15 | Multi-Class ROC Curve | [View](project.pdf#page=11) |
| Fig. 16 | Cordana ROC | [View](project.pdf#page=12) |
| Fig. 17 | Healthy ROC | [View](project.pdf#page=12) |
| Fig. 18 | Pestalotiopsis ROC | [View](project.pdf#page=12) |
| Fig. 19 | Sigatoka ROC | [View](project.pdf#page=12) |
| Fig. 20–23 | Grad-CAM Visualizations | [View](project.pdf#page=13) |

---

## 👨‍💻 Research Authors

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

## 🛠️ Technical Concepts

- Artificial Intelligence
- Deep Learning
- Convolutional Neural Networks
- Spatial Attention Mechanisms
- Image Preprocessing
- Image Augmentation
- Image Classification
- Residual Connections
- Batch Normalization
- Global Average Pooling
- Softmax Activation
- Confusion Matrix
- Classification Metrics
- ROC-AUC
- Grad-CAM
- Explainable AI

---

## 🌾 Potential Applications

The research can support future development in:

- AI-assisted agricultural disease identification
- Banana plantation monitoring
- Plant disease research
- Automated crop health analysis
- Computer vision in agriculture
- Farmer decision-support tools
- Explainable agricultural AI systems

---

## ✅ Conclusion

This research presents a Convolutional Neural Network integrated with a custom KernelAttention spatial attention mechanism for automated banana leaf disease classification.

The model classifies four banana leaf conditions: Healthy, Sigatoka, Cordana Leaf Spot, and Pestalotiopsis Leaf Blight.

Using the BananaLSD dataset, the proposed method achieved **97.50% test accuracy**.

Its performance was evaluated using multiple classification metrics, confusion matrix analysis, ROC curves, and AUC measurements.

Grad-CAM visualizations were used to investigate the image regions influencing model predictions.

The study provides a foundation for further research into attention-based deep learning models for agricultural disease diagnosis.

---

## 📬 Contact

**Koushik Daggupati**

**GitHub:** [DGSKOUSHIK](https://github.com/DGSKOUSHIK)

**Repository:** [Banana-Leaf-Disease-Classification](https://github.com/DGSKOUSHIK/Banana-Leaf-Disease-Classification)

---

<p align="center">
  <b>🍌 Banana Leaf Disease Classification Using CNN and KernelAttention</b>
</p>

<p align="center">
  <i>Deep Learning | Computer Vision | Explainable AI | Smart Agriculture</i>
</p>
