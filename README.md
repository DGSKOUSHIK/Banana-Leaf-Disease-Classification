
# 🍌 Automated Detection and Classification of Banana Leaf Diseases Using CNN with Spatial Attention Mechanism

<p align="center">
  <img src="https://img.shields.io/badge/Python-Deep%20Learning-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-CNN-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Computer%20Vision-Image%20Classification-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Research-Agricultural%20AI-purple?style=for-the-badge" />
</p>

<p align="center">
  <b>An attention-enhanced Convolutional Neural Network for accurate and interpretable banana leaf disease classification.</b>
</p>

<p align="center">
  <a href="#-project-overview">Overview</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-proposed-architecture">Architecture</a> •
  <a href="#-experimental-results">Results</a> •
  <a href="#-grad-cam-explainability">Explainability</a>
</p>

---

## 📌 Project Overview

Banana is an economically important agricultural crop that supports food security and provides income to farmers, particularly in tropical and subtropical regions.

However, banana production is seriously affected by fungal leaf diseases, which reduce photosynthetic activity, damage leaves, and negatively affect crop yields.

Traditional disease identification depends on manual inspections by agricultural specialists. Such methods can be time-consuming, expensive, and difficult to scale to large farming environments.

This research proposes a **Convolutional Neural Network (CNN) integrated with a custom spatial attention mechanism called KernelAttention** to automatically classify banana leaf images into four categories.

### Supported Classes

1. **Healthy**
2. **Sigatoka**
3. **Cordana Leaf Spot**
4. **Pestalotiopsis Leaf Blight**

The model learns disease-specific visual patterns such as lesions, discoloration, necrotic regions, and irregular spots.

Unlike a conventional CNN, the proposed architecture includes spatial attention modules that selectively emphasize important regions in the feature maps.

The model achieved **97.50% classification accuracy on a test dataset containing 760 images**.

---

## 🎯 Research Objectives

The primary objectives of this research are:

- Develop a deep learning system for automated banana leaf disease classification.
- Identify three major banana leaf diseases and distinguish them from healthy leaves.
- Design a CNN architecture integrated with a custom spatial attention mechanism.
- Improve disease feature extraction through hierarchical attention.
- Address class imbalance using image augmentation.
- Evaluate the model using accuracy, precision, recall, specificity, F1-score, and ROC-AUC.
- Analyze prediction errors using a confusion matrix.
- Provide interpretable predictions using Grad-CAM.
- Explore the potential of computer vision in agricultural disease diagnosis.

---

## 🌿 Banana Leaf Disease Categories

### 1. Healthy Banana Leaves

Healthy banana leaves generally have consistent green coloration without significant disease-related lesions.

Correct identification of healthy leaves is necessary to distinguish normal foliage from disease symptoms and avoid unnecessary interventions.

![Healthy Banana Leaves](assets/fig01.png)

### 2. Sigatoka Disease

Sigatoka is a major fungal banana leaf disease associated with dark spots and streaks on leaf surfaces.

**Common visual symptoms:**

- Small dark-brown or black spots
- Elongated streaks along leaf structures
- Yellowing around infected regions
- Enlarging lesions
- Drying and necrosis of infected tissue

Advanced infections reduce photosynthetic activity and can seriously affect banana production.

![Sigatoka Disease](assets/fig02.png)

### 3. Cordana Leaf Spot

Cordana Leaf Spot, caused by *Cordana musae*, produces characteristic leaf lesions.

**Common visual symptoms:**

- Circular or irregular spots
- Water-soaked patches
- Brown lesions
- Drying of infected tissue
- Premature deterioration of leaves

The disease can reduce functional leaf area and contribute to lower plant productivity.

![Cordana Leaf Spot](assets/fig03.png)

### 4. Pestalotiopsis Leaf Blight

Pestalotiopsis Leaf Blight is associated with irregular brown lesions and necrotic regions.

**Common visual symptoms:**

- Irregular brown patches
- Water-soaked areas
- Expanding lesions
- Merging necrotic regions
- Damaged leaf surfaces

Its visual similarities to other fungal diseases can make automated classification challenging.

![Pestalotiopsis Leaf Blight](assets/fig04.png)

---

## 📊 Dataset

### BananaLSD Dataset

The research uses the **BananaLSD dataset**, consisting of banana leaf photographs captured under natural farming conditions in Bangladesh.

The original dataset contains **937 RGB images** covering the four classification categories.

Images were collected using smartphone cameras and standardized to a size of **224 × 224 pixels** for model processing.

Unlike datasets captured in controlled laboratory conditions, field images contain natural variations in lighting, background, viewing angle, and image composition.

These variations make the classification task more realistic.

### Original Dataset Composition

| Disease Category | Original Images | Augmented Images |
|---|---:|---:|
| Pestalotiopsis Leaf Blight | 173 | 400 |
| Sigatoka | 473 | 400 |
| Cordana Leaf Spot | 162 | 400 |
| Healthy | 129 | 400 |
| **Total** | **937** | **1600** |

### Final Experimental Dataset

| Dataset Split | Number of Images |
|---|---:|
| Training | 1,351 |
| Validation | 120 |
| Testing | 760 |
| **Total** | **2,231** |

The test dataset consists of 190 images per class.

The dataset preparation aims to maintain balanced evaluation and reduce the influence of majority classes during training.

### Dataset Distribution

![Dataset Distribution](assets/fig05.png)

---

## 🔄 Data Preprocessing and Augmentation

Image preprocessing prepares the leaf photographs for CNN training and improves consistency between samples.

### Image Preprocessing

- Resize images to 224 × 224 pixels.
- Process RGB images with three color channels.
- Organize data into four classification categories.
- Prepare separate training, validation, and test subsets.

### Data Augmentation

The research employs several image transformations to improve sample diversity.

| Technique | Purpose |
|---|---|
| Random rotation | Handles variation in leaf orientation |
| Horizontal flipping | Increases orientation diversity |
| Width shifting | Simulates horizontal position changes |
| Height shifting | Simulates vertical position changes |
| Shearing | Represents viewing-angle distortions |
| Zooming | Simulates differences in camera distance |
| Fill mode | Handles empty regions introduced by transformations |

The purpose of augmentation is to reduce overfitting and help the CNN learn meaningful disease characteristics despite differences in image acquisition.

---

## 🧠 Proposed Architecture

The proposed deep learning model consists of an initial CNN feature extraction block, three KernelAttention blocks, and a final classification head.

### Architecture Pipeline

```text
              Banana Leaf Image
                (224 × 224 × 3)
                       |
                       v
              Conv2D (32 Filters)
                       |
                       v
               Batch Normalization
                       |
                       v
                 ReLU Activation
                       |
                       v
                 MaxPooling2D
                       |
                       v
             KernelAttention Block
                  (64 Filters)
                       |
                       v
                 MaxPooling2D
                       |
                       v
             KernelAttention Block
                 (128 Filters)
                       |
                       v
                 MaxPooling2D
                       |
                       v
             KernelAttention Block
                 (256 Filters)
                       |
                       v
                 MaxPooling2D
                       |
                       v
             Global Average Pooling
                       |
                       v
                Dense (64, ReLU)
                       |
                       v
                  Dropout (0.5)
                       |
                       v
                Dense (4, Softmax)
                       |
                       v
                Disease Prediction
```

### Model Architecture Diagram

![CNN with KernelAttention Architecture](assets/fig06.png)

### Architecture Components

#### Initial Convolution Layer

The first convolution layer contains 32 filters of size 3 × 3.

The purpose of this layer is to detect low-level features such as edges, textures, boundaries, and simple visual patterns.

#### Batch Normalization

Batch normalization helps stabilize activations during training.

#### ReLU Activation

The Rectified Linear Unit introduces nonlinearity:

`ReLU(x) = max(0, x)`

#### Max Pooling

Max pooling reduces spatial feature-map dimensions and computational requirements.

A pool size of 2 × 2 is used in the architecture.

#### KernelAttention Blocks

Three consecutive attention blocks use:

- 64 filters
- 128 filters
- 256 filters

These blocks progressively learn more complex features while applying spatial attention.

#### Global Average Pooling

Global average pooling converts learned spatial feature maps into compact feature representations.

#### Dense Classification Head

The final classification stage consists of:

- Dense layer with 64 units and ReLU activation
- Dropout with rate 0.5
- Dense output layer with four units and Softmax activation

Softmax produces class probabilities for the four supported categories.

---

## 🔍 KernelAttention: Proposed Spatial Attention Mechanism

The main architectural contribution of this research is a custom spatial attention mechanism named **KernelAttention**.

Traditional CNNs learn spatial features through convolution operations, but not every region of a leaf image contributes equally to disease identification.

A disease may occupy a relatively small region of a leaf while the remaining image contains healthy tissue or background objects.

KernelAttention aims to increase the influence of diagnostically relevant regions in the learned feature representation.

### KernelAttention Internal Architecture

![KernelAttention Internal Architecture](assets/fig07.png)

### Working Principle

The attention block creates three learned feature representations:

- Query (Q)
- Key (K)
- Value (V)

These representations are generated using convolutional transformations.

### Step 1: Query Generation

`Q = ReLU(BN(Conv3×3_Q(X)))`

The query representation captures learned visual features from the incoming feature map.

### Step 2: Key Generation

`K = ReLU(BN(Conv3×3_K(X)))`

The key representation provides another learned description of the image features.

### Step 3: Value Generation

`V = ReLU(BN(Conv3×3_V(X)))`

The value representation contains the features to be weighted by the attention map.

### Step 4: Attention Map Calculation

The query and key representations are combined:

`A_intermediate = Q + K`

A 1 × 1 convolution and sigmoid function then generate the spatial attention map:

`A = sigmoid(Conv1×1(A_intermediate))`

The sigmoid function maps attention weights to values between 0 and 1.

### Step 5: Applying Attention

The attention map is multiplied element-wise with the value representation:

`C = V ⊙ A`

Here, `⊙` represents element-wise multiplication.

This allows different locations in the feature maps to receive different attention weights.

### Step 6: Residual Connection

The attention-weighted representation is combined with the original or projected input:

`Y = C + X_projected`

If the input and output channel dimensions are different, a 1 × 1 convolution projects the input to the required dimensions.

This residual connection preserves information from earlier feature representations.

### Advantages of KernelAttention

1. Emphasizes spatially important image features.
2. Supports the identification of disease-associated lesion patterns.
3. Integrates attention at multiple architectural depths.
4. Combines attention-weighted and residual information.
5. Supports hierarchical feature learning.
6. Provides an architecture suitable for later visual explanation using Grad-CAM.

---

## ⚙️ Model Training

The proposed model was evaluated using a training process lasting **40 epochs**.

Training performance was monitored through accuracy and loss measurements.

### Training Accuracy

![Model Training Accuracy](assets/fig08.png)

The model demonstrated increasing training accuracy during the initial epochs.

At the final stage of training, the reported performance reached:

| Metric | Result |
|---|---:|
| Test Accuracy | 97.50% |
| Validation Accuracy | 99.17% |
| Number of Epochs | 40 |

Training accuracy approached 100% in the final epochs.

### Training Loss

![Model Training Loss](assets/fig09.png)

The loss graph displays the training, validation, and test loss curves.

The reported curves generally decreased during model training.

This indicates that the model reduced prediction errors as the training process progressed.

---

## 📈 Experimental Results

The trained CNN with spatial attention was evaluated using:

- Classification accuracy
- Precision
- Recall / sensitivity
- Specificity
- F1-score
- Confusion matrix
- Receiver Operating Characteristic (ROC) curves
- Area Under the ROC Curve (AUC)
- Grad-CAM visualizations

### Overall Test Accuracy

# 97.50%

The model correctly classified **741 out of 760 test images**, according to the reported confusion matrix.

### Per-Class Classification Performance

| Disease | Accuracy | Precision | Recall | Specificity | F1-Score |
|---|---:|---:|---:|---:|---:|
| Cordana | 99.34% | 99.47% | 97.89% | 99.82% | 98.67% |
| Healthy | 98.68% | 98.91% | 95.79% | 99.65% | 97.33% |
| Pestalotiopsis | 97.76% | 93.03% | 98.42% | 97.54% | 95.65% |
| Sigatoka | 99.21% | 98.94% | 97.89% | 99.65% | 98.41% |

### Cordana Performance

![Cordana Classification Metrics](assets/fig10.png)

Cordana classification achieved:

- Accuracy: 99.34%
- Precision: 99.47%
- Recall: 97.89%
- Specificity: 99.82%
- F1-score: 98.67%

### Healthy Leaf Performance

![Healthy Leaf Classification Metrics](assets/fig11.png)

Healthy leaf classification achieved:

- Accuracy: 98.68%
- Precision: 98.91%
- Recall: 95.79%
- Specificity: 99.65%
- F1-score: 97.33%

### Sigatoka Performance

![Sigatoka Classification Metrics](assets/fig12.png)

Sigatoka classification achieved:

- Accuracy: 99.21%
- Precision: 98.94%
- Recall: 97.89%
- Specificity: 99.65%
- F1-score: 98.41%

### Pestalotiopsis Performance

![Pestalotiopsis Classification Metrics](assets/fig13.png)

Pestalotiopsis classification achieved:

- Accuracy: 97.76%
- Precision: 93.03%
- Recall: 98.42%
- Specificity: 97.54%
- F1-score: 95.65%

The lower precision of the Pestalotiopsis class indicates that some images from other categories were incorrectly predicted as Pestalotiopsis.

---

## 🧮 Confusion Matrix Analysis

A confusion matrix describes the number of correct and incorrect predictions for each disease category.

![Confusion Matrix](assets/fig14.png)

### Test Confusion Matrix

| Actual / Predicted | Cordana | Healthy | Pestalotiopsis | Sigatoka |
|---|---:|---:|---:|---:|
| Cordana | 186 | 0 | 4 | 0 |
| Healthy | 0 | 182 | 7 | 1 |
| Pestalotiopsis | 0 | 2 | 187 | 1 |
| Sigatoka | 1 | 0 | 3 | 186 |

### Observations

**Cordana:** 186 of 190 samples were correctly classified.

**Healthy:** 182 of 190 samples were correctly classified. Seven were predicted as Pestalotiopsis and one as Sigatoka.

**Pestalotiopsis:** 187 of 190 samples were correctly classified.

**Sigatoka:** 186 of 190 samples were correctly classified.

The confusion matrix shows that most samples are located on the main diagonal, representing correct predictions.

The most frequent error involves predicting Healthy images as Pestalotiopsis.

---

## 📉 ROC Curves and AUC Analysis

Receiver Operating Characteristic curves measure how well a classification model distinguishes positive and negative examples across different decision thresholds.

The ROC curve plots:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

The Area Under the Curve (AUC) summarizes the discriminatory performance.

### Multi-Class ROC Analysis

![Multi-Class ROC Curve](assets/fig15.png)

The research reports:

| Measurement | AUC |
|---|---:|
| Micro-Average AUC | 0.999 |
| Macro-Average AUC | 0.999 |
| Cordana AUC | 0.999 |
| Healthy AUC | 0.999 |
| Pestalotiopsis AUC | 0.998 |
| Sigatoka AUC | 1.000 |

### Cordana ROC Curve

![Cordana ROC Curve](assets/fig16.png)

**AUC: 0.999**

The ROC result indicates strong discrimination between Cordana and the remaining classes.

### Healthy ROC Curve

![Healthy ROC Curve](assets/fig17.png)

**AUC: 0.999**

The reported result indicates strong discrimination of healthy leaves from diseased samples.

### Pestalotiopsis ROC Curve

![Pestalotiopsis ROC Curve](assets/fig18.png)

**AUC: 0.998**

This class has the lowest reported AUC among the four categories, although its discriminatory performance remains high on the test dataset.

### Sigatoka ROC Curve

![Sigatoka ROC Curve](assets/fig19.png)

**AUC: 1.000**

The reported Sigatoka ROC-AUC indicates perfect ranking discrimination on the evaluated test set.

This does not mean every Sigatoka image was classified correctly at the selected decision threshold.

---

## 🔥 Grad-CAM Explainability

Deep learning systems can provide accurate predictions while offering limited visibility into how their decisions are made.

To support interpretability, this project uses **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM produces heatmaps that identify image regions contributing to the model's predicted class.

### Why Grad-CAM?

Grad-CAM helps investigate whether the trained CNN is focusing on relevant disease symptoms or unrelated image regions.

For banana leaf disease classification, important features can include:

- Dark fungal spots
- Brown leaf lesions
- Disease-related discoloration
- Necrotic patches
- Vein-oriented streaks
- Changes in leaf texture

### Grad-CAM Visualizations

#### Cordana Grad-CAM

![Cordana Grad-CAM](assets/fig20.png)

The heatmap can be used to inspect the regions influencing Cordana predictions.

#### Healthy Leaf Grad-CAM

![Healthy Leaf Grad-CAM](assets/fig21.png)

This visualization helps investigate the regions influencing predictions of healthy foliage.

#### Sigatoka Grad-CAM

![Sigatoka Grad-CAM](assets/fig22.png)

The research discusses attention patterns associated with leaf veins and disease-related streaks.

#### Pestalotiopsis Grad-CAM

![Pestalotiopsis Grad-CAM](assets/fig23.png)

This visualization helps examine the image regions influencing Pestalotiopsis predictions.

### Explainability Benefits

- Visual inspection of influential image regions
- Improved understanding of model behavior
- Identification of possible background bias
- Support for model debugging
- Better communication of predictions to researchers

Grad-CAM provides evidence about model attention, but a heatmap alone does not establish that every highlighted region is a medically or botanically correct disease indicator.

---

## 🆚 Comparison With Related Research

The research paper reviews several previously published approaches for banana leaf and plant disease detection.

| Method | Reported Accuracy |
|---|---:|
| BananaSqueezeNet | 96.25% |
| Eight-Layer DCNN | 98.92% |
| DenseNet-201 | 98.12% |
| VGG16 | 97.81% |
| ResNet50 | 94.00% |
| VGG19 | 94.00% |
| TGVFCMS + CNN | 93.45% |
| ANN + SIFT | 95.00% |
| **Proposed CNN + KernelAttention** | **97.50%** |

**Important:** These results originate from different studies with potentially different datasets, class definitions, splits, and experimental conditions.

Therefore, the table is a literature summary and should not be treated as a controlled, identical-dataset benchmark.

The main architectural distinction of this project is its custom KernelAttention mechanism with hierarchical spatial attention.

---

## 🏆 Key Contributions

### 1. Custom KernelAttention Mechanism

The research introduces a spatial attention module combining convolutional query, key, and value representations with a residual connection.

### 2. Hierarchical Spatial Attention

Three attention blocks progressively extract more complex disease-related image features.

### 3. Four-Class Disease Classification

The model predicts Healthy, Sigatoka, Cordana, and Pestalotiopsis classes.

### 4. Dataset Balancing

Image augmentation helps address the imbalance in original disease categories.

### 5. Detailed Experimental Evaluation

Performance is examined using classification metrics, ROC curves, AUC scores, and a confusion matrix.

### 6. Visual Interpretability

Grad-CAM visualizations provide a way to inspect which regions influence model predictions.

---

## ⚠️ Limitations

Although the reported model demonstrates strong classification performance, several limitations remain.

### Dataset Size

The original dataset contains 937 images, which limits the diversity of observed conditions.

### Geographic Diversity

The dataset was collected in a specific geographic region. Performance may differ in other countries or growing environments.

### Image Quality

Poor lighting, motion blur, occlusions, and low-resolution images may affect predictions.

### Similar Disease Symptoms

Some fungal diseases produce visually similar leaf lesions, which can cause misclassification.

### Class Coverage

The classifier supports four categories and does not cover all banana diseases, pests, or nutrient deficiencies.

### Real-World Validation

Additional evaluation across independent farms, camera devices, and unseen environmental conditions is needed before making broad deployment claims.

### Model Comparison

A controlled ablation experiment comparing the same CNN with and without KernelAttention would be useful for measuring the contribution of the attention mechanism.

---

## 🚀 Future Enhancements

Potential extensions of the research include:

1. Developing a web application for uploading banana leaf images.
2. Deploying the model through a mobile application.
3. Expanding the dataset to additional banana diseases.
4. Introducing disease severity estimation.
5. Supporting disease localization or segmentation.
6. Evaluating lightweight architectures for edge devices.
7. Testing robustness against different lighting and camera conditions.
8. Performing ablation experiments on KernelAttention blocks.
9. Comparing CNN-based attention with modern vision transformer approaches.
10. Conducting field trials with independent agricultural datasets.

These are proposed future directions, not completed features.

---

## 📁 GitHub Repository Structure

```text
Banana-Leaf-Disease-Classification/
│
├── README.md
│
├── project.pdf
│
└── assets/
    ├── fig01.png
    ├── fig02.png
    ├── fig03.png
    ├── fig04.png
    ├── fig05.png
    ├── fig06.png
    ├── fig07.png
    ├── fig08.png
    ├── fig09.png
    ├── fig10.png
    ├── fig11.png
    ├── fig12.png
    ├── fig13.png
    ├── fig14.png
    ├── fig15.png
    ├── fig16.png
    ├── fig17.png
    ├── fig18.png
    ├── fig19.png
    ├── fig20.png
    ├── fig21.png
    ├── fig22.png
    └── fig23.png
```

The structure above organizes the research documentation and its visual assets. The model source code and trained weights must be added separately if they are available.

---

## 📄 Research Paper

The complete research paper is available in this repository.

### [📖 Read the Full Research Paper](project.pdf)

**Title:**

Automated Detection and Classification of Banana Leaf Diseases Using Convolutional Neural Network With Spatial Attention Mechanism

The paper describes the dataset, proposed architecture, mathematical formulation, experiments, performance metrics, and Grad-CAM visualizations.

---

## 👨‍💻 Authors

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

## 🛠️ Technologies and Concepts

The project is based on:

- Deep Learning
- Convolutional Neural Networks
- Computer Vision
- Spatial Attention
- Image Classification
- Image Augmentation
- Batch Normalization
- Residual Connections
- Softmax Classification
- ROC-AUC Evaluation
- Confusion Matrix Analysis
- Grad-CAM Explainability

---

## 🌾 Applications

Potential applications include:

- Automated agricultural disease screening
- AI-assisted banana plantation monitoring
- Research in plant pathology
- Agricultural computer vision
- Early-stage disease classification
- Decision-support systems for farmers
- Explainable artificial intelligence in agriculture

These are potential application areas and do not imply that a production-ready agricultural system has been deployed.

---

## ✅ Conclusion

This project investigates the classification of banana leaf diseases using a CNN integrated with a custom KernelAttention spatial attention mechanism.

The proposed model was evaluated on a four-class banana leaf dataset and achieved **97.50% reported test accuracy**.

Its strongest reported classification results include high precision and recall across the disease categories, near-perfect ROC-AUC measurements, and interpretable Grad-CAM heatmaps.

The proposed approach provides a research foundation for developing computer vision-based agricultural disease screening systems.

Further work is needed to establish generalization on independent datasets and evaluate the contribution of the attention mechanism through controlled experiments.

---

## 📬 Contact

**Koushik Daggupati**

GitHub: [DGSKOUSHIK](https://github.com/DGSKOUSHIK)

Repository: [Banana-Leaf-Disease-Classification](https://github.com/DGSKOUSHIK/Banana-Leaf-Disease-Classification)

---

<p align="center">
  <b>Automated Banana Leaf Disease Classification Using Deep Learning and Spatial Attention</b>
</p>

<p align="center">
  Research Project | Computer Vision | Agricultural Artificial Intelligence
</p>
