# EEG Stress Detection Using Machine Learning 

### Project Overview 

This project aims to detect stress levels from EEG(Electroencephalography) signals using various machine learning algorithms. By analyzing brainwave patterns, we build models that classify whether an individual is stressed or not. The project leverages feature extraction techniques like Continuous Wavelet Transform (CWT), Fast Fourier Transform (FFT), and Discrete Wavelet Transform (DWT) to enhance model accuracy.

### Novelty 

The unique aspect of this project lies in the incorporation of Continuous Wavelet Transform (CWT) as a feature extraction method alongside traditional techniques such as Fast Fourier Transform (FFT) and Discrete Wavelet Transform (DWT). CWT provides superior time-frequency resolution, allowing for more detailed analysis of EEG signals.

- CWT: Captures both time and frequency components of EEG signals, enabling better localization of transient changes in the data.
- FFT: Provides frequency domain analysis, but with limited time localization.
- DWT: Offers multi-resolution analysis, which is effective in decomposing EEG signals into different frequency bands, but without the continuous nature of CWT.

### Dataset 
- Source: [Download_here] (https://www.sciencedirect.com/science/article/pii/S2352340921010465)
- The dataset contains 32 Channel EEG Time-series data labeled with stress levels, including:
    - Tasks:
      - Stroop Color Word Test
      - Mirror Image Recognition Task
      - Arithmetic Problem Solving Task
    - Features: EEG signals across multiple frequency bands (Theta, Alpha, etc.)
    - Target: tress levels categorized into classes (1–10), with 1–6 representing low to moderate stress and 7–10 representing higher stress.
### Methods 
We apply various machine learning techniques to detect stress, with a focus on advanced signal processing for feature extraction:
  1. Data Preprocessing:
     - Filtering and noise reduction of raw EEG signals.
     - Feature extraction using DWT, CWT, and FFT to capture time-frequency information.
  2. Modeling:
     - Machine Learning Models: k-Nearest Neighbour(kNN), Long Short-Term Memory(LSTM), and Support Vector Machine(SVM).
  3. Evaluation:
     - Metrics: Accuracy, Precision, Recall, F1 Score
     - Confusion matrix analysis
    
### Usage 
1. Preprocessing:
   
Before training models, preprocessing steps are required to filter, clean, and extract features from the EEG data. Each notebook applies **Principal Component Analysis (PCA)** after feature extraction (using CWT, DWT, or FFT) to reduce dimensionality and prepare the data for classification models.

You can run the preprocessing notebooks for each case using the following commands in Jupyter Notebook:

 **- Preprocessing after CWT:**
   - Preprocessing PCA after CWT-KNN.ipynb
   - Preprocessing PCA after CWT-LSTM.ipynb
   - Preprocessing PCA after CWT-SVM.ipynb
   
 **- Preprocessing after DWT:**
   - Preprocessing PCA after DWT-KNN.ipynb
   - Preprocessing PCA after DWT-LSTM.ipynb
   - Preprocessing PCA after DWT-SVM.ipynb

 **- Preprocessing after FFT:**
   - Preprocessing PCA after FFT-KNN.ipynb
   - Preprocessing PCA after FFT-LSTM.ipynb
   - Preprocessing PCA after FFT-SVM.ipynb
   
2. Experiment Cases
After preprocessing, each combination of feature extraction method and machine learning model is tested in a separate experiment case. The notebooks below run the experiments and evaluate model performance.

 **Continuous Wavelet Transform (CWT):**
   - Experiment Case of CWT + KNN .ipynb
   - Experiment Case of CWT + SVM .ipynb
   - Experiment Case of CWT-LSTM.ipynb

 **Discrete Wavelet Transform (DWT):**
   - Experiment Case of DWT + KNN .ipynb
   - Experiment Case of DWT + SVM .ipynb
   - Experiment Case of DWT-LSTM.ipynb

 **Fast Fourier Transform (FFT):**
  - Experiment Case of FFT + KNN .ipynb
  - Experiment Case of FFT + SVM .ipynb
  - Experiment Case of FFT + LSTM.ipynb

Running a specific experiment:

Open the relevant experiment notebook (e.g., Experiment Case of CWT + KNN .ipynb).
Load the preprocessed data and run the machine learning model.
View the results including accuracy, confusion matrices, and other performance metrics.

### Results
The key results are:
- DWT + kNN achieved the highest accuracy of 97.0%.
- CWT + LSTM produced an accuracy of 92.6%, marking the lowest performance.
- The models performed best at detecting low to moderate stress levels (classes 1–6), with reduced accuracy in higher stress levels (classes 7–10).

### Contributing
If you would like to contribute to this project, feel free to submit a pull request or open an issue with suggestions.
