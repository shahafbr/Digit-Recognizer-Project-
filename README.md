# Digit Recognizer Project

This repository contains a project aimed at building machine learning models to recognize handwritten digits. The project employs several machine learning techniques, comparing their performance in digit classification.

## Project Structure

- **Dataset**: The training data is stored in `Dataset_Train.csv`.
- **Jupyter Notebook**: The core implementation and analysis are in `Digit_Recognizer.ipynb`.
- **Models**: Pre-trained models are stored in the `models/` directory.

## Data Preparation

The dataset consists of 1,530 samples, each represented by 3,025 pixel values (55x55 images) and a corresponding label (digit 0-9). The data preparation process includes:

1. **Loading the Data**: The dataset is loaded into a pandas DataFrame.
2. **Visualization**: Random samples from the dataset are visualized to understand the structure of the images.
3. **Data Splitting**: The dataset is split into training and testing sets to evaluate model performance.

## Model Training

### 1. **Support Vector Classifier (SVC)**

- **Algorithm**: Support Vector Machines (SVM) are used for classification, finding the optimal hyperplane that separates different digit classes.
- **Training**: The SVC model is trained using the pixel data from the training set.
- **Results**: The model achieved an accuracy of `0.996732` on the test set.

### 2. **Random Forest**

- **Algorithm**: The Random Forest algorithm builds multiple decision trees and merges them to obtain a more accurate and stable prediction.
- **Training**: The Random Forest model is trained on the training dataset, leveraging multiple trees to improve generalization.
- **Results**: The model achieved an accuracy of `1.000000` on the test set (replace with actual results).

### 3. **Multilayer Perceptron (MLP)**

- **Algorithm**: A Multilayer Perceptron (MLP) is a type of artificial neural network that consists of multiple layers of neurons with a fully connected structure.
- **Training**: The MLP model is trained using the pixel data from the training set, employing backpropagation for learning.
- **Results**: The MLP model achieved an accuracy of `0.993464` on the test set (replace with actual results).

### 4. **Convolutional Neural Network (CNN) using Keras**

- **Algorithm**: A deep learning model, specifically a Convolutional Neural Network (CNN), is used for feature extraction and classification.
- **Training**: The CNN model is trained on the training dataset with layers designed to capture spatial hierarchies in the image data.
- **Results**: The CNN model achieved an accuracy of `0.996732` on the test set (replace with actual results).

## Testing and Evaluation

Each model is evaluated on the test set, with performance metrics such as accuracy being the primary measure of success. The following summarizes the performance:

- **SVM Model**: Accuracy - `0.996732`
- **Random Forest Model**: Accuracy - `1.000000`
- **MLP Model**: Accuracy - `0.993464`
- **CNN Model**: Accuracy - `0.996732`

These results highlight the strengths and weaknesses of different machine learning algorithms in the context of digit recognition, with deep learning models typically offering higher accuracy.

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/digit-recognizer.git
   cd digit-recognizer
2. **Install the necessary dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook Digit_Recognizer.ipynb
4. **Train and Test the Models**:
   - Follow the steps in the notebook to train and evaluate the models.
##Conclusion
This project demonstrates the application of various machine learning algorithms, including SVM, Random Forest, CNN, and MLP, in recognizing handwritten digits. The comparison of these models provides insights into their relative performance and suitability for image classification tasks.

