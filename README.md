# -Stock-Prediction-Using-VADER-Sentiment-CNN-Based-Subjectivity-and-LSTM
# in sentiment analysis using vader method, we can not distinguish between subjective opinions and objective facts in the headlines.To address this, I use a CNN model trained on the Cornell dataset to extract subjectivity and objectivity (SubjObj) scores for each headline.

we know that the drawback of the VADER lexicon is that it cannot determine whether a sentence is subjective or objective, meaning it can’t tell if the sentence is a fact or an opinion. before finding the subjectivity and objectivity of the headlines it is better to define cornell dataset:
# Cornell Subjectivity Dataset
The Cornell Subjectivity Dataset was created by Bo Pang and Lillian Lee, two researchers from Cornell University. It was developed as part of their research in the field of subjectivity and sentiment analysis in natural language processing (NLP). they collected a dataset with 5,000 subjective and 5,000 objective sentences from IMDb
The Cornell Subjectivity Dataset is a labeled collection of short English texts  designed to train and evaluate models for subjectivity classification, for distinguishing subjective language from objective language.

in this project we first prepared a labeled dataset using the Cornell Subjectivity Dataset. This dataset contains two pre-annotated text corpora:
•	quote.tok.gt9.5000: a collection of 5,000 subjective sentences, typically drawn from movie review excerpts expressing personal opinions or emotions.
•	plot.tok.gt9.5000: a collection of 5,000 objective sentences, primarily consisting of plot summaries that state facts without sentiment.
Each sentence in the subjective set was assigned a binary label of 0, and each in the objective set received a label of 1. These labels reflect the ground truth classification for supervised learning. We then concatenated the two datasets into a single NumPy array X containing 10,000 sentences, and a corresponding array y of 10,000 binary labels. This unified dataset forms the foundation for training the subjectivity/objectivity classifier. This step is crucial as it transforms raw text into a supervised learning-compatible format, where each sentence is paired with a ground truth label. The resulting (X, y) pair is then ready for further preprocessing, tokenization, and vectorization using pre-trained embeddings prior to model training. 
Each sentence is labeled as:
•  0 for subjective content,
•  1 for objective content.
These 10,000 labeled sentences were used to train and evaluate our deep learning model.
for classification of each sentence as subjective or objective, we aim to implement 1D Convolutional Neural Network (CNN) but we know that neural networks require numerical input, the 10,000 labeled sentences that we have are not numerical so we will use word embedding method to convert each sentence into a vector.
before that i want to define the worwmbedding and introduce GloVe word embedding method.
## Word embedding is a technique used in Natural Language Processing (NLP) to convert words into numerical vectors so that machines can understand and work with human language. 
Common Pre-trained Embeddings:
•	GloVe (Global Vectors)
•	Word2Vec
•	FastText
•	ELMo, BERT (contextual embeddings) 
Glove has pre-defined dense vectors for around every 6 billion words of English literature along with many other general-use characters like commas, braces and semicolons. The algorithm's developers frequently make the pre-trained GloVe embeddings available
Users can select a pre-trained GloVe embedding in a dimension like 50-d, 100-d, 200-d or 300-d vectors that best fits their needs in terms of computational resources and task specificity. 

in this project we have chosen GloVe because GloVe is a widely used pre-trained embedding model that captures semantic relationships between words.
•	Each word was mapped to a 300-dimensional vector from the glove.6B.300d.txt embedding.
•	Sentences were tokenized and padded/truncated to a fixed maximum length of 40 tokens.
•	Words not found in the GloVe vocabulary were assigned a random "unknown word" vector.
This produced consistent, fixed-length numerical inputs for all sentences.

the we implement a 1D Convolutional Neural Network (CNN) in PyTorch to classify each sentence as subjective or objective.
* CNN 
CNN is a specific class of deep neural networks and perhaps the most popular algorithm among the deep learning environments. 
The typical structure of a CNN is composed of three layers: convolutional, pooling, and fully connected layers. 
One-dimensional convolutional neural networks (1D-CNNs) have been around since the late 1980s, Generally, 1D-CNNs are designed to handle one-dimensional data, such as time-series data, sequences (e.g., text), or any data where the primary structure is along a single axis. The kernel (or filter) in a 1DCNN moves along one dimension.
*Conv1D (3,4,5):	Three convolution layers with kernel sizes 3, 4, and 5, 100 filters each
*Activation:	ReLU activation to introduce non-linearity
*MaxPooling: Extracts most prominent features from each convolutional path
*Dropout:	Dropout with probability 0.5 to prevent overfitting
*Fully Connected:	Linear layer with sigmoid activation for binary classification
Once we had all 10,000 labeled sentences ready (subjective and objective), we split them into three parts:
1.	Training Set (80%) – This is used to train the model so it can learn from examples.
2.	Validation Set (10%) – This is used during training to check how well the model is performing on unseen data and to fine-tune the model’s settings.
3.	Test Set (10%) – This is used only once at the end to evaluate the final performance of the model.
we ensure the model is trained, validated, and tested on separate data to avoid overfitting and get an honest measure of performance.
# Stock Market Movement Prediction (Classification)
•	Each news day is scored with average Subj/Obj classification across the 25 headlines.
we Merge Sentiment & Stock Data
All sentiment scores (compound, neg, pos, neu, and SubjObj_Score) are merged with financial indicators (Open, High, Low, Volume, etc.)
________________________________________
Final Features Used for prediction (classification of stock up/down):

['compound', 'neg', 'pos', 'neu', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'SubjObj_Score']
________________________________________
Classification Models are
XGBoost	Ensemble model, Random Forest, Logistic Regression, K-Nearest Neighbors, Decision Tree, Gaussian Naive Bayes
Each model is evaluated using:
•	Train/Test accuracy
•	Classification report (precision, recall, F1)
•	Confusion matrix

# Regression part: Stock Price Prediction Using LSTM 
In this part we focus on forecasting the numerical stock price value (regression task) using a Long Short-Term Memory (LSTM) neural network. The objective is to predict the Adjusted Close price of the Dow Jones Industrial Average (DJIA) index based on a combination of financial indicators and sentiment-related features extracted from news headlines.

We used a merged dataset that combines, Stock market data (Open, High, Low, Close, Volume, Adj Close), Sentiment scores from VADER (compound, neg, pos, neu) and Subjectivity score from a CNN trained on the Cornell Subjectivity Dataset (SubjObj_Score), the target variable is Adj Close

* Preprocessing
The dataset was split chronologically into training and testing sets based on the date (train < 2015-01-01, test ≥ 2015-01-01) 
MinMaxScaler was used to normalize the features and target values.
We then created time-series sequences of length 10 days. Each input sample to the LSTM model consists of 10 consecutive days of features, and the target is the adjusted closing price on the following day.

* LSTM Model Architecture
A deep LSTM model was implemented using TensorFlow Keras:
Three LSTM layers with 128, 64, and 32 units respectively
Dropout layers after each LSTM to prevent overfitting
A final Dense layer for regression output
The model was trained using Mean Squared Error (MSE) as the loss function and the Adam optimizer
Training was performed for up to 200 epochs with early stopping (patience = 20) based on the validation loss to avoid overfitting.

Evaluation Metrics
After training, the model's performance was evaluated on the test set using the following regression metrics:
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score
* Mean Absolute Percentage Error (MAPE)

Additionally, the model's predicted stock prices were plotted against the actual prices to visually assess prediction performance.

* Results
The model achieved a relatively low error and a good R² score on both training and test sets, indicating that it was able to capture patterns in the historical and sentiment data to predict future stock prices.
The loss curves for training and validation showed convergence and no major signs of overfitting.









