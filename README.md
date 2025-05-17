# LLM-Generated-Text-Detection-Using-Machine-Learning

## Overview
This project implements a **Logistic Regression** based machine learning pipeline to detect whether a piece of text is **human-generated** or **AI-generated** by large language models (LLMs). It uses TF-IDF vectorization and standard machine learning preprocessing steps, training, evaluation, and inference. The goal is to provide a reliable and interpretable method to distinguish AI-generated text.

## Dataset

The datasets used are publicly available from Kaggle:

- **LLM-Detect AI Generated Text (DAIGT)**  
  [https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset/data](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset/data)

- **Augmented Dataset for LLM-Detect**  
  [https://www.kaggle.com/datasets/sunilthite/augmented-data-for-llm-detect-ai-generated-text](https://www.kaggle.com/datasets/sunilthite/augmented-data-for-llm-detect-ai-generated-text)

> Please download these datasets and place them in the `/data` folder or update the data path in the notebook before running.


## Features

- Text preprocessing including cleaning and balancing of classes
- TF-IDF vectorization using unigrams and bigrams
- Logistic Regression model training and evaluation
- Performance metrics: Accuracy, Precision, Recall, F1-score, and Confusion Matrix visualization
- Model serialization using `pickle` for saving and loading
- Sample code for inference on custom input text



## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/areebaghazal88/LLM-Generated-Text-Detection-Using-Machine-Learning.git
   cd LLM-Generated-Text-Detection-Using-Machine-Learning
   ```

2. (Optional) Create and activate a Python virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## Usage

1. **Prepare the Dataset**
   Download and place the Kaggle datasets in the `/data` directory.

2. **Run the notebook or script**
   Open `LLM_Text_Detection.ipynb` in your preferred Jupyter environment (Google Colab, Jupyter Notebook, VSCode, etc.) to:

   * Load and preprocess the dataset
   * Train the Logistic Regression model
   * Evaluate the model and visualize metrics

3. **Make Predictions**
   Example for inference on new text:

   ```python
   import pickle

   # Load model and vectorizer
   with open('logistic_regression_model.pickle', 'rb') as f:
       logreg = pickle.load(f)
   with open('tfidf_vectorizer.pickle', 'rb') as f:
       tfidf_vectorizer = pickle.load(f)

   # Sample text prediction
   text = "Once upon a time in a forest, a little girl met three bears..."
   vectorized = tfidf_vectorizer.transform([text])
   prediction = logreg.predict(vectorized)
   label = "human-generated" if prediction == 0 else "AI-generated"
   print("Prediction:", label)
   ```



## Dependencies

* Python 3.x
* scikit-learn
* pandas
* numpy
* seaborn
* matplotlib
* joblib

Install them via:

```bash
pip install -r requirements.txt
```

## GUI Interface

A screenshot of the graphical user interface (GUI) is included in this repository as `gui_image.png`.  
The GUI provides a user-friendly way to interact with the AI-generated text detection system, allowing users to input text and receive real-time detection results.



## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.


Thank you for checking out this project! Contributions and suggestions are welcome.

