# Sentiment Based Text Classification on Amazon Reviews Using BER

This project performs sentiment analysis on the Amazon Polarity dataset using a fine-tuned BERT transformer model.
It’s developed as part of a research methods assignment focusing on training, fine-tuning, and evaluating large language models (LLMs) for text classification tasks.

---

###  Dataset

- **Name:** Amazon Polarity
- **Description:** Large-scale dataset of Amazon product reviews labeled as positive or negative.
- **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/mteb/amazon_polarity)

---

### Tools and Libraries 

- Python 
- Google Colab 
- matplotlib
- PyTorch
- Hugging Face Transformers – For loading pre-trained BERT models and fine-tuning
- Hugging Face Datasets – For downloading and handling the Amazon Polarity dataset
- scikit-learn – For computing metrics like accuracy, precision, recall, and F1-score




###  Methodology

- Exploratory Data Analysis (EDA) on review text and label distribution.
- Text cleaning and tokenization using BERT’s tokenizer.
- Fine-tuning of a pre-trained BERT model for binary sentiment classification.
- Manual hyperparameter tuning with different learning rates.
- Evaluation using accuracy, F1-score, precision, and recall.
- Visualization of training progress and evaluation metrics.
- Saving and loading the trained model for inference.


### License

This project is licensed under MIT License.Please see the license tab 

---

