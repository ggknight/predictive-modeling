Performance Assessment 

Gretchen J. Knight 

Western Governor’s University 

D208: Predictive Modeling

Contents

Performance Assessment

# Part I: Research Question

## A1: Research Question

What factors predict the variable Additional Charges, indicating the average amount charged to the patient for miscellaneous items, on top of the total charge? These charges include procedures, treatments, medicines, anesthesiology, and other things.

## A2: Goals

The objective of my analysis is to gain greater insight to determine which medical and lifestyle factors directly correlate with the amount of a patient’s average additional charges. The cost of medical treatments and hospital stays are not transparent here in the United States, and a model like this could help with informed patients and a more accountable healthcare system. To accomplish these goals, I will perform multiple linear regression to identify the factors that influence additional charges.

# Part II: Method Justification

## B1: Summary of Assumptions

Multiple linear regression includes several main assumptions. First is the assumption that there is a linear relationship between the dependent variable and each of the independent variables. This can be verified through a scatter plot. Second is the assumption that the predictor variables are not too highly correlated with each other. We want these predictor variables to vary independently (as the name suggests) and correlation between these variables can cause multicollinearity. Third is the assumption that each observation is selected randomly and is independent from the other observations. Each data point should only be represented once and should not depend on the results of another observation. Fourth and finally, linear regression assumes that residuals are normally distributed around a mean of zero. This indicates that the level of error is random across the full range of data (Zach, Statology 2021). 

## B2: Tool Benefits 

I chose to use Python for this analysis. This choice was driven by the professor recommending sticking with the programming language I had used for my previous classes. Additionally, I learned how to do multiple linear regression using R in a course during my undergraduate degree, and I wanted to practice using Python this time. As described in the R or Python webpage found on Western Governors University’s website (2023), “Python’s focus on readability and simplicity makes its learning curve relatively linear and smooth.” I appreciate the clear code resources for creating multiple linear regression using Python that are found online. For transparency’s sake, I think this task may have been simpler to have completed in R. R is specifically made by statisticians, and the code is often more straightforward. I am happy with the new skills I learned in Python though. The packages that I used are pandas (allows for easier data manipulation), NumPy (computations), Matplotlib, Plotly Express and Seaborn (for graphic and visual representations of the data), and StatsModels (VIF to check for multicollinearity and SM for linear regression and other analyses) (Western Governors University, 2023).

## B3: Appropriate Technique

Multiple linear regression is an appropriate approach to find which factors influence additional charges for medical patients because we are looking at many independent variables and trying to create a model to predict the response variable. Using MLM allows us to find how much of an impact the predictor variables have on the response and gives us an idea of how minor changes in the predictors can affect the outcome.

# Part III: Data Preparation

## C1: Data Cleaning

To clean the data, I began by looking at the information frame of the dataset to verify that all values were non-null. There were no null values. I continued by looking at the dependent variable and the thirteen independent variables of interest. I was specifically looking for outliers. For each of the variables, the data is valid and does not have outliers that are of concern. The values outside the 'typical range' are still acceptable data points. I did not treat for outliers. The following code gives an example of what I did for the different continuous and categorical variables to check for null values and outliers. For the complete cleaning, see the attached code.
```python
medical.info()

medical['Age'].describe()

fig = px.histogram(medical, x='Age', nbins=30, title = 'Age')

fig.show()

medical['HighBlood'].value_counts()
```
## C2: Summary Statistics

The following shows the results of exploratory summary statistics on each of the independent and dependent variables. For the quantitative variables, I used .describe() to get the basic statistics. These include mean (which describes the average value), standard deviation (describing how spread out the observations are), median (denoted as 50% which is the middle observation), and range (as described by the minimum and maximum values). For the qualitative variables, I used .value_counts() to get a summary. Qualitative values do not have a mean or standard deviation, but you can view the proportion of each answer. All variables had a count of 10,000 observations. For each quantitative variable below, I will highlight the mean and median and how they compare; for qualitative variables, I will provide percentages out of 10,000 observations.

## 

#### C3: Visualizations

Univariate Visualizations

Bivariate Visualizations

## C4: Data Transformation

My goal for data transformation was to create dummy variables for all qualitative factors. Some of these factors only had two levels of responses, while others had multiple response options. There is a process called one-hot encoding that creates new columns for each category of the variable. Pandas has a function (.get_dummies()) that will create new columns (often called dummy variables). Within the function, I specified to drop the first value. This creates k -1 variables, as to prevent multicollinearity. Additionally, I specified the data type for the dummy variables at ‘int64’. This gives the output in 0 and 1, which allows for ease in calculations, the main point of doing this transformation. The following code shows selecting the variables and the specifications as noted above. Please see the attached code for more details. 
```python
dummy_medical = pd.get_dummies(medical[['Area', 'Marital', 'Gender', 'ReAdmis',

'Soft_drink', 'Initial_admin', 'HighBlood', 'Stroke', 'Overweight',

'Complication_risk', 'Arthritis', 'Diabetes', 'Hyperlipidemia', 'BackPain', 'Anxiety', 'Allergic_rhinitis', 'Reflux_esophagitis', 'Asthma', 'Services']], drop_first=True, dtype = 'int64')
```
## C5: Prepared Data Set

	See the attached data set: “medicalprepared.csv”

# Part IV: Model Comparison and Analysis

## D1: Initial Model

Additional Charges = - 2910.84 + 225.75 * (Age) + 8630.70  * (HighBlood_Yes) – 289.38 * (Complication_risk_Low) + 360.89 * (Stroke_Yes) + 461.25 * (Initial_admin_Emergency) + 0.00006 * (Total Charge) – 107.51 * (Initial_admin_observation) + 43.54 * (Marital_Married) + 44.86 * (Marital_Never Married) + 16.61 * (Full_meals_eaten) – 9.68 * (Area_Suburban) – 1.91 * (Allergici_rhinitis_Yes) – 32.37 * (BackPain_Yes)

## D2: Justification of Model Reduction

Now that I have my initial model, I need to reduce the number of features to improve the overall model significance. For this reduction, I elected to use Backward Stepwise Elimination, a wrapper method. Wrapper methods look at the features in the model. Then, based on model performance, you add or remove specific features. In Backward Stepwise Elimination, you “start with all the features and remove the least significant feature (based on p-value) at each iteration which improves the performance of the mode” (Middleton 2022). I iterated through this process until all features had significant p-values (p < 0.05).

## D3: Reduced Linear Regression Model

Additional Charges = - 2891.94 + 225.73 * (Age)  +  8631.39  * (HighBlood_Yes)  –  291.16 * (Complication_risk_Low) + 360.56 * (Stroke_Yes) + 460.96 * (Initial_admin_Emergency) – 108.33 * (Initial_admin_observation)

## E1: Model Comparison

The following table shows a comparison of the values used to compare and evaluate models. There is not a major difference between the initial and reduced models regarding these evaluation metrics. Adjusted R-squared signifies the variation in the response variable that is explained by the explanatory variables. The adjusted R-squared was high initially and improved minutely by 0.00002. The F-statistic has the largest improvement of all the metrics. It improves from 11570.23 to 25076.15 which is a difference of 13505.92. However, both the initial and reduced models have highly significant F-statistics. AIC penalizes for errors made with new variables; BIC compares goodness of fit. In both cases, the model with the lower value implies a better model. AIC and BIC both decrease in the reduced model. Finally, Residual Standard Error (RSE) measures the standard deviation of the residuals. The value indicates the error. In this case, the model would predict additional charges with an average error of $1633. The reduced model shows a miniscule improvement of $0.2591.

## E2: Output and Calculations

The Residual Standard Error (RSE) for the reduced model is 1633.27. This indicates that the average error for predicted additional charges is $1633.27. 

The following plots show the residual plot of residuals versus predicted values. As shown, there is an obvious pattern. You want your residual plots to not have a visible pattern, so this is a bit of a problem. In the next plot, I plot the same data and filter by whether the patient has high blood pressure. Like what I saw earlier in this analysis, blood pressure is affecting these results. This would be interesting to look at in a later course.

Below is the quantile-quantile plot, commonly referred to as a Q-Q plot. This s-shape curve informs us that the data is under-dispersed. This means that the data has a reduced number of outliers (the tails are thinner than expected). Since we are looking at residuals, this indicates that our residuals may not be randomly distributed, as previously discovered with the residual plot (Yearsley 2024). 

## E3: Code

See attached code.

# Part V: Data Summary and Implications

## F1: Results

Regression Equation:

Additional Charges = - 2891.94 + 225.73 * (Age)  +  8631.39  * (HighBlood_Yes)  –  291.16 * (Complication_risk_Low) + 360.56 * (Stroke_Yes) + 460.96 * (Initial_admin_Emergency) – 108.33 * (Initial_admin_observation)

Interpretation of coefficients:

For every year older the patient is, additional charges increase by $225.73.

If the patient has high blood pressure, additional charges increase by $8631.39.

If the patient has a low risk for complications, additional charges decrease by $291.16.

If the patient had a stroke, additional charges increase by $360.56.

If the patient was initially admitted in an emergency, additional charges increase by $460.96.

If the patient was initially admitted for observation, additional charges decrease by $108.33.

Significance:

	This model has statistical significance. The adjusted R-squared is high, showing that the variation in the dependent variable is explained by the independent variables. Additionally, the p-value for the F-statistic is much less than 0.001, showing that the regression results are not likely to be due to chance. These results should be easily repeatable. 

	This model has less practical significance. These results could be helpful to both patients and billing professionals to know what additional charges a patient may accrue from their hospital stay. The patient’s age, blood pressure, complication risk, stroke, and initial admission type are all easily gathered and can be entered into the regression equation quickly and at the start of a hospital stay. However, several of the assumptions for linear regression were violated in this model. Running the correlation between the independent variables and dependent variable only showed a high linear correlation between age and blood pressure as compared to additional charges with 0.71 and 0.65 correlation coefficients, respectively. The remaining other variables had a correlation of less than 0.05. For the sake of running a model, I left them in the equation, but their impact is not large. Additionally, as seen above, the residuals were not normally distributed. Something was happening with high blood pressure regarding all the other variables. The results are limited based on that. Ideally, you would sort the data into high/low blood pressure, and then run separate analyses on those groups, as the patterns are different. I suspect that separating the groups would allow more of the assumptions to be met, giving more practical significance. 

Limitations:

	Once again, the results of this analysis are limited due to the high blood pressure variable causing issues. Further, I did not eliminate any outliers. All values appeared acceptable to me, so I decided to leave them in. Regression is not resistant to outliers, and this may have had a negative effect. Additionally, correlation does not mean causation. In the case of this model, a patient may think they know what their additional charges will be based on the criteria, and they could ultimately be surprised by a difference in charges. In my model reduction phase, I only used backward stepwise elimination. There were mentions of RFE (recursive feature elimination), but I did not run that in this analysis. It would be interesting to separate the groups by high blood pressure status, check the correlation between variables, and run the regression models again.

## F2: Recommendations

Returning to the research question, I was wondering what factors influence additional charges. My findings indicate that age and high blood pressure are the greatest contributing factors, with complication risk, stroke, and initial admission type all having a small influence as well. Based on all these results and considerations of significance and limitations, this model is not yet significant, but it could be with a few minor adjustments. I would first split the data set into two groups: those with high blood pressure and those without high blood pressure. I would run the analyses again, this time using RFE and verifying all model assumptions as part of the analysis. This model showing additional charges a patient may incur would be helpful to both those within the hospital administration and patients. Cost transparency in the medical field is something that can use vast improvements, and this model could be a first step in helping achieve that.

# Part VI: Demonstration

# Third-Party Code References

Sewell, William. (2023, July 13). D208 Webinar Ep. 1. [PowerPoint slides]. Masters of Science, Data Analytics, Western Governors University. 

# References

Middleton, Keiona. (2022, November 7). Getting Started with D208 Part I. [PowerPoint slides]. Masters of Science, Data Analytics, Western Governors University. 

Yearsley, John. (2024, January 1). QQ Plots – An Overview. University College Dublin. 

Zach. (2021, November 16). The five assumptions of multiple linear regression. Statology.  