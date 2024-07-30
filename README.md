# Hotel Recommendation System

Hotel RS is an autoencoder-based hotel recommendation system trained on the TripAdvisor dataset. The system aims to provide personalized hotel recommendations to users.

## Project Structure

- `dataset/`: Contains the TripAdvisor dataset used for training and testing the recommendation system.
- `index.ipynb`: Jupyter notebook with full code.

## Phases

### Phase 1: Dataset Preparation
#### Dataset
The TripAdvisor dataset, sourced from GitHub, includes six columns: '<b>review id</b>', '<b>member id</b>', '<b>hotel id</b>', '<b>rating</b>', '<b>recommend list</b>', and '<b>review text</b>'. The '<b>recommend list</b>' column is of particular interest as it contains criteria-specific ratings in the format <b>{rating:criteria}</b> (ex: ”4:rating; 5:Cleanliness; 3:Location; 3:Rooms; 4:Service; 5:Sleep
Quality; 4:Value”). The dataset features <b>3453 users</b>, <b>1832 hotels</b>, and <b>21826 reviews</b>.

#### Pre-processing
To prepare the dataset for training and testing, several pre-processing steps were performed:
1. The '<b>recommend list</b>' column was parsed to extract individual ratings, which were then added as new columns in a dataframe, filling non-existent ratings with <b>NaN</b> values to ensure completeness for each metric.
2. Records that included '<b>checkin</b>' and '<b>business service</b>' ratings were eliminated due to inconsistency, reducing the dataset by <b>800 records</b>.
3. Duplicate records were removed based on '<b>member id</b>' and '<b>hotel id</b>' columns to ensure uniqueness.
4. The '<b>review id</b>', '<b>review text</b>', and '<b>recommend list</b>' columns were dropped as they were not necessary for the analysis.

| Metric | Org dataset | Processed Dataset |
|--|--|--|
| Users | 3453 | 3446 |
| Hotel | 1832 | 1832 |
| Records | 21826 | 19600 |
| Max hotels rated by a user | 24 | 24 |
| Min hotels rated by a user | 1 | 1 |
| Avg hotels rated by a user | 5.91 | 5.69 | 

#### Train-Test split
The dataset was divided into a training set and a testing set. <b>NaN</b> values were replaced with <b>0</b>s, and only entries with non-zero values in all fields were selected, reducing the number of hotels to <b>1811</b>. A test split of <b>5%</b> was implemented, ensuring all users in the test set were excluded from the training set for fair model evaluation. A <b>user×hotel</b> matrix was generated for the training set. For testing, two matrices were created: y_test, containing user×hotel matrices filled with 0s except for test set records, and x_test, containing full user records with test set values set to 0.
| Array | Dimension | Non 0 values |
|--|--|--|
| X_TRAIN | 2641x1811 | 12217 |
| X_TEST | 729x1811 | 3772 |
| Y_TEST | 729x1811 | 818 |

### Phase 2: Autoencoder Construction
#### AutoEncoder 
1. <b>Defining the Autoencoder Class</b>: Using PyTorch, the autoencoder class is defined with parameters such as layers, sizes, and activation functions.
2. <b>Weight Initialization</b>: Initializing weights from a normal distribution with mean 0 and standard deviation 0.02 is found to perform better than Xavier initialization, bias is set as 0.
3. <b>Loss Function</b>: Mean Squared Error (MSE) is chosen as the metric for reconstruction loss to evaluate the dissimilarity between the input and output of the autoencoder.
4. <b>Regularisation</b>: To prevent overfitting L2 regularisation is used with a weight decay of 0.1.
5. <b>Parameters</b>: Learning rate of 0.001, with batch size = 100 and num epochs = 200 is used.

### Phase 3 - Train Test and Comparison
#### Training
A total of 7 autoencoders were trained with specified parameters and evaluated. Gradients were accumulated after each iteration instead of being zeroed,
for faster and improved model convergence.
#### Evaluation Metrics
We use Root Mean Squared Error (RMSE) for assessing the performance of these models.
An additional comparison is made: every hotel is rated as 2.5 for each user, as non-intelligent recommendations and then compared with our models' RMSE.
This comparison highlights the value of the neural network’s intelligence in making more accurate and personalized recommendations.

#### Testing
During the testing phase, each model is evaluated by passing the test data (<b>x_test[criteria]</b>) through the corresponding model (<b>model[criteria]</b>) and calculating the RMSE for each model with respect to <b>y_test[criteria]</b> for all seven criteria. For the predicted overall rating, the collaborative filtering principle is applied by averaging all <b>y_pred[criteria]</b> and comparing it with <b>y_test[rating]</b> (overall rating).

## Hyperparameter validation 
Results are highlighted in the tables below:
| Number of criteria | RMSE - calculated | RMSE - 2.5 |
|--|--|--|
| 1 | 0.98 | 1.68 |
| 7 | 0.975 | 1.68 |

| Number of encoder layers | RMSE - calclulated | RMSE - 2.5 |
|--|--|--|
| 1 | 0.975 | 1.68 |
| 3 | 0.977 | 1.68 |
| 5 | 1.008 | 1.68 |
| 7 | 1.008 | 1.68 |

| Inner activation function | Outer activation function | RMSE - calculated | RMSE - 2.5 |
|--|--|--|--|
| Sigmoid | ELU | 0.975 | 1.68 |
| Sigmoid | Linear | 0.9751 | 1.68 |
| Tanh | ELU | 1.005 | 1.68 |
| Softmax | ELU | 3.06 | 1.68 |

| Optimization | RMSE - calculated | RMSE - 2.5 | 
|--|--|--|
| RMSProp | 0.975 | 1.68 |
| Adam | 1.004 | 1.68 | 
| Adagrad | 1.254 | 1.68 | 
| SGD | 2.38 | 1.68 |

## Model comparison
| Algorithm | RMSE - calculated |
| --| -- |
| Autoencoder | 0.975 |
| Multi Criteria User based Collaborative Filtering | 1.051 | 

## Acknowledgements
Q. Shambour, A deep learning based algorithm for multi-criteria recommender systems,
Knowledge-Based Systems (2020)

