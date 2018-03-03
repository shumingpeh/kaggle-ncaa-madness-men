# NCAA March Madness Kaggle Competition

Link: [https://www.kaggle.com/c/mens-machine-learning-competition-2018]

## Instructions for Tim
```
git clone https://github.com/shumingpeh/kaggle-ncaa-madness-men.git
```
Msg Shuming to add you as a contributor.

Before you start working on the project, `git pull`

If you're making a change that you're confident will not critically impact the work Shuming/I are concurrently doing, push your changes directly to master. `git push origin master`

If you suspect you're making a critical change, first create a new branch. `git checkout -b <name>`
Then push the changes of that branch. `git push origin <name>`
And log unto github to submit a Pull Request to merge your branch with master.

## To-do 
~~1. Download and explore the dataset~~
~~2. Identify the features we want to use in our model.~~
~~- So far we've shortlisted:~~
    ~~- Seeding~~
    ~~- Home/Away Court~~
~~3. Build a simple model and test it against Kaggle hold out~~

1. Baseline accuray of ~70%, objective to hit >= 90% accuracy
2. Replicate ROC curve from our logistics data to output
3. Add in more features
4. Feature selection
	- Correlation matrix: @ddaniel
	- RFE for logistics regression
	- AIC for logisitcs regression
	- Variance Threshold for logistics regression
5. How do we prevent overfitting? What are the precautionary measures to ensure we are not overfitting?
6. Testing out prediction with different models
	- Random forest
	- SVM
	- DL
	- Ridge Regression
	- XGBoost