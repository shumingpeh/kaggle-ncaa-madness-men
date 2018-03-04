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

~~1. Baseline accuray of ~70%, objective to hit >= 90% accuracy~~
~~2. Replicate ROC curve from our logistics data to output~~
~~3. Add in more features~~
	~~- who is coaching~~
	~~- home court~~
	~~- Region~~
	~~- all the diff variables or variables to use to throw into our initial dataframe~~
~~4. Feature selection~~
	~~- Correlation matrix: @ddaniel~~
	~~- RFE for logistics regression: @ddaniel~~
	~~- AIC for logisitcs regression: @shumingpeh~~
	~~- Variance Threshold for logistics regression: @shumingpeh~~
~~5. How do we prevent overfitting? What are the precautionary measures to ensure we are not overfitting? @shumingpeh @ddaniel~~
~~6. Testing out prediction with different models~~
	~~- Random forest~~
	~~- SVM~~
	~~- DL~~
	~~- Ridge Regression~~
	~~- XGBoost~~

1. Add in all features that will be used in our final model
	- season year
	- region
	- winning teamid
	- winning score
	- losing teamid
	- losing score
	- winning field goal percentage
	- winning three point percentage
	- winning free throw percentage
	- losing field goal percentage
	- losing three point percentage
	- losing free throw percentage
2. Add in all intermediate variables that will be used in our final model
	- winning and losing team offensive and defensive rebounds
	- winning and losing team assits
	- winning and losing team steals
	- winning and losing team blocks
	- winning and losing team personal fouls
3. Feature selection
	- Correlation matrix
	- RFE for logistics regression
	- AIC for logistics regression
	- Variance threshold for logistics regression
4. How do we prevent overfitting? What are the precautionary measures to ensure we are not overfitting?
5. Testing out predictin with different models
	- RF
	- SVM
	- DL
	- Ridge regression
	- XGBoost

## Timeline
- Adding in features: 6th March
- Adding in intermediate variables: 6th March
- Feature selection/prevent overfitting: 8th March
- Testing out with different models: 10th March