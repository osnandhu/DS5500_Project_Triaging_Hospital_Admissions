Course : DS 5500 54510 Information Visualisation
Professor In-charge : Andrew Therriault

PROJECT 1: Triaging Hospital Admission
PROJECT PARTNERS : Govind Bhala, Nanditha Sundararajan

INTRODUCTION:

In hospitals, the term triage refers to sorting of injured or sick people according to their need for emergency medical attention. It is a method of determining priority for who gets care first.
Triage is used when the medical-care system is overloaded, meaning there are more people who need care than there are available resources to care for them
Triage is performed by nurses after examining patients when they first arrive in order to place them in one of the following categories:
	1. ADMIT- Those patients who need immediate advanced medical assistance (Eg. Accidents, Surgeries)
	2. DISCHARGE - Those patients who require minimal medical assistance (Eg. Fever,Consultations)
These assessments are done under time critical environments. Delayed triage is a huge liability risk for physicians and can directly be correlated with poorer patient outcomes.

Thus we aim to apply machine learning  to ribustly triage incoming patients based on the available information and patient history.
Early identification of patients who are likely to require immediate admission may enable better optimization of hospital. Faster assessment and notification to administrators regarding potential admissions may help alleviate this problem.
A patient's likelihood of admission may serve as a proxy for acuity, which is used in several downstream decisions such as bed placement and the need for emergency intervention.

DATASET:
	1. Data/TriagingDataset.csv - It is a 560,486 x 972 dataframe. It is de-identified according to the norms. Save the dataframe in a local folder for exporting it in Jupyter Notebook for further analysis.

SETUP:
This project requires the following software packages:
1.Jupyter Notebook (Anaconda with Python 3.x) or Google Colab
2.The dependencies are included in the dependencies_triage.txt file.
	To install the packages use :
	pip install -r dependencies_triage

Files in Scripts/ :
1. Triage_script.ipynb - This is a notebook which has step by step implementation of preprocessing( EDA, Imputation.Dataset 
				   reduced to ~560,486 x 743),memory reduction of the dataframe,Test and Train split up for model training (XGBoost, RandomForest),GridSearch for hyperparameter tuning to obtain the optimal models and calculation of feature importance.
				   

2. Neuralnets_triage.ipynb - This is a notebook where a neural model has been been trained on the preprocessed 	    
							 datataframe with the top 300 features obtained by sorting the feature importances


3. app.py - 

Files in models/
1. Xgb_reg.pkl- Pickled file of the optimal XGBoost model that gave the best results post hyperparameter tuning.
2. opt_rf_model - Pickled file of the optimal Random Forest model that gave the best results post hyperparameter tuning.
3. nn_opt_model.h5 - Keras optimal Neural Network model that gave the best results.

Files in Result_data/
1. final_df - Pickled file in form of a dictionary of dataframes that can be used for model training. The optimised	
			  dataframe was split as X_train,X_test in the ratio 80:20.Since the original data set was too large to be processed using a Non GPU based system, we dropped training data points in 3 options, keeping 70% , 60% and 50% of train rows. In this process, we didn't disturb test set at all. Test set contains all rows (20% of data) which were split from the original data set.
				
			 The intent of using test train split to drop data points was to ensure the distribution of positive and negative cases is still the same in each such combination of train data.

			 The final_df contains:
			 a)X_train_a - 70% of X_train set is retained
			 b)X_train_b - 60% of X_train set is retained
			 c)X_train_c - 50% of X_train set is retained
			 d)Y_train_a - 70% of Y_train set is retained
			 e)Y_train_b - 60% of Y_train set is retained
			 f)Y_train_c - 50% of Y_train set is retained
			 g) X_test - test set (20% of the original dataset)
			 h) Y_test - test set of the target variable 


2. feature_imp.pkl - A pickle file which consists of the 743 features and their respective importances arranged in 							 descending order

CONCLUSION:
For robustly predicting the triage , we preprocessed the dataset and ran various classification models like XGBoost, Random Forest and a deep learning model. We did try running adaboost but since it cannot be parallelized, it took a lot of time or the time efficiency of the algorithm was very less when compared to XGBoost and RandomForest.
We have evaluated the best model based on the FBeta score (Beta = 2). Precision is twice as important as Recall in this case as we want to minimise the False Negatives. In other words, we dont want our model to classify a patient who is in a serious condition as 'Discharge' which will in-turn decrease the credibility of our model.

1)The results of our best XGBoost model obtained post Hyperparameter tuning(max_depth = 15,n_estimators = 40,learning rate = 0.2):
AUC : 0.9208
FBeta score: 0.9269
Confusion Matrix: [[2981 10486]
					[4665,73966]]  



2)The results of our best RandomForest model obtained post Hyperparameter tuning(max_depth = 15,n_estimators = 75):
AUC : 0.90422
FBeta score: 0.9369
Confusion Matrix: [[19395 14072]
					[2851,75780]]  


2)The results of our best Neural Network model obtained post Hyperparameter tuning(no. of nodes in first layer = 60):
AUC : 0.8108	
FBeta score: 0.9203
Confusion Matrix: [[23075 10392]
					[5328,73303]]  


Thus, from the above results, we can see that the false negatives is significantly less in Random Forest when compared to the other models. We can also observe that there is a trade-off between AUC score and F-Beta score. Though the Random Forest's AUC score is less than that of the XGBoost, it performs better by significantly reducing the false negatives. Thus, our model wont misclassify a patient who is in a serious condition as 'Discharge'.


CITATIONS:
1. https://github.com/yaleemmlc/admissionprediction
2.https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016
3. Medium.com
4. TowardsDataScience.com
5. https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

					

			 
