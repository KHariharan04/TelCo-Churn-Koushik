# ==============================================================================

# MAST6100 Final Project: TelCo Customer Churn Risk Analysis
# A project by - Koushik Hariharan, kh630@kent.ac.uk
# Module: Machine Learning and Deep Learning - MAST6100/AUT
# Final Version for submission
# ==============================================================================

# ------------------------------------------------------------------------------
# SECTION 0: PRELUDE - LIBRARIES AND SETUP
# ------------------------------------------------------------------------------
# RStudio version used is 2025.09.2+418
# R version used is 4.5.2 (2025-10-31 ucrt)


install.packages(c("ggplot2","glmnet","performance","ISLR","leaps","pROC",
"e1071","corrplot"))

install.packages("caret",dependencies = TRUE) 

# The Caret package is the classification and regression training package in R
# which are all crucial for our project here.
# We are loading all packages right at the outset here, 
# because these are constantly deployed throughout the project.
# Installing and loading them here makes it easier for us to deploy these on the
# go, as opposed to having to attach them every time.


# Loading the packages now
library(ggplot2)  # For visualisation, version 4.0.0
library(caret)    # For data partitioning and confusion matrices, version 7.0.1
library(e1071)    # For skewness and other stats, version 1.7.16
library(pROC)     # For ROC curve analysis, version 1.19.01
library(MASS)     # For AIC and BIC, version 7.3.65
library(glmnet)   # For Penalised Regression, version 4.1.10
library(class)    # For KNN, version 7.3.23
library(corrplot) # For correlation visualisation, version 0.95
library(nnet)     # For neural networks, version 7.3.20


# Setting a seed to ensure all results are reproducible
set.seed(10)


# ------------------------------------------------------------------------------
# PRELUDE: EXPLANATION OF WHICH DATASET WE WILL USE AND WHAT IT ACTUALLY IS
# ------------------------------------------------------------------------------

# The dataset we will be working on is the TelCo Churn Dataset
# This dataset is the data of the customers of a Telecom company.
# Each row represents one customer (unique, more on this later)
# and each column is a feature/characteristic associated with the customer.

# The data-set can be downloaded from the link below, if needed:
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Our aim here is to predict 'Churn'. Churn occurs when a user is no longer 
# a customer. It is essentially equal to terminating the services with the
# company.

# Understanding this is very important because it drives the company's
# strategies, including marketing, sales, pricing, product cycle etc.

# Some of the features/characteristics we will study include
# Who the customer is. There are a few parameters associated with this.

# They include gender, senior citizen status, whether or not
# they have any dependents, whether or not they have a partner, etc.
# These capture the churn odds with respect to the customer's status.
# General examples that we test for include:
# Customers with dependents may be less likely to switch providers
# Seniors may be more conservative

# The next set of characteristics include
# Price and related details, and their relationship with the firm
# These include details of the contract, like contract tenure, contract type,
# payment method, billing type (paperless or not)

# These help us analyse churn odds with respect to switching costs.
# Example - long tenure customers may be less likely to switch, whereas
# month-to-month customers may be more likely to switch.
# Those who are on auto-pay may also be less likely to switch.

# The next set of characteristics include
# Internet Service, Tech Support, Streaming TV, etc.
# These capture what services they use with the company, how the access
# the internet etc.

# They tell us how many services of the firm they use.
# If one uses more services, they may be more embedded in the company's
# ecosystem and may be unlikely to switch, and so on.

# The final set of characteristics include financial related ones, such as
# MonthlyCharges, TotalCharges.

# These capture where the revenue comes from, and subsequently, where revenue
# may be lost. High monthly values with low corresponding total charges means
# higher risk of churn and so on.

# The sections of this coding project are

# SECTION 0 - PRELUDE (We are here now)
# SECTION 1 - DATA CLEANSING AND EXPLORATORY DATA ANALYSIS
# SECTION 2 - LOGISTIC REGRESSION MODELS, PARAMETER TUNING AND MODEL SELECTION
# SECTION 3 - PENALISED REGRESSION MODELS - RIDGE AND LASSO REGRESSION
# SECTION 4 - K NEAREST NEIGHBOURS
# SECTION 5 - NEURAL NETWORKS (DEEP LEARNING)
# SECTION 6 - CONCLUSIONS AND FINAL REMARKS

# The code here is extensively commented and is self-contained, however, we have
# also prepared reports and power-point presentations to summarise the findings
# from this exercise and to summarise the high-level takeaways succinctly.
# These are contained in the Moodle submission.

# ==============================================================================
# SECTION 1: EDA BEGINS! DATA LOADING AND INITIAL INSPECTION
# ==============================================================================

# Load the dataset
# Firstly, please ensure the working directory is set to where 
# the dataset file is located
churn_data <- read.csv("WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                       stringsAsFactors = TRUE)
# The stringAsFactors function is very convenient, enabling us to convert
# all the categorical covariates into factors in one go, rather than doing
# so manually.


# Initial inspection of the data structure
# Dataset Dimensions
dim(churn_data)
# As expected, it has 7043 rows and 21 columns, which is what we expected.

#First 5 rows of the data, just so we understand what we are working with:
head(churn_data)

#Structure of the dataset in question, after the factor conversion earlier
str(churn_data)
# The summary shows us that all the covariates which should have been converted 
# into factors have been done so, thanks to the one line in read.csv()


# Summary statistics for all variables to identify initial issues
summary(churn_data)
# Initial inspection of the summary statistics shows no issues and
# demonstrates coherence in the data, which is ideal and what we'd expect.



# ------------------------------------------------------------------------------
# DATA CLEANING AND PRE-PROCESSING
# ------------------------------------------------------------------------------

# Firstly, we need to check if the customer_ID column has any repeated values.
# Let us do so.

sum(duplicated(churn_data$customerID))

# The output here is zero, which means the customerID column is a unique 
# identifying column. This tells us that each row MUST correspond to a unique
# customer's value.

# Furthermore, note the following below:
# CustomerID is unique to each row and is insignificant for
# our purpose, since we are focused only on classification and testing.

# Having already identified that each row corresponds to a unique customer,
# we note that the customerID column is rather redundant.
# Keeping it is suboptimal from a coding point of view, hence we remove it.

churn_data$customerID <- NULL # It has been removed

# Handling Missing Values (Data Quality Check)
# Checking for NA values in the entire dataset
sum(is.na(churn_data)) # As we can see, we have only 11 missing values

sum(is.na(churn_data))/nrow(churn_data)
# The error rate is 0.001561834, which is 0.1562% approximately.
# This project has a leeway of upto 30% missing values,
# but our churn_data falls way below this 30% gap, which is very
# convenient for our purpose.

# Investigating where these missing values are occurring
na_counts <- colSums(is.na(churn_data))
na_counts[na_counts > 0]
# 'TotalCharges' often contains blanks when the customers are new,
# and this is the reason the values are repeated.

# Check if all repeated values are from this column
sum(is.na(churn_data$TotalCharges))
# Yes, the same number 11 is repeated here, so all repeated values
# are indeed from the same column.

# How do we deal with the missing data/values?

# Since these are likely new customers who haven't generated a bill yet,
# it is safe to assume their TotalCharges are 0. 
churn_data$TotalCharges[is.na(churn_data$TotalCharges)] <- 0

# Verify that missing values are indeed removed.
(sum(is.na(churn_data)) == 0)
# TRUE is generated, which means that there are no more missing values anymore.

# 3.3 correcting Data Types
# The SeniorCitizen variable appears to be integer but should be a factor 
# (Category)
churn_data$SeniorCitizen <- as.factor(ifelse(churn_data$SeniorCitizen == 0, 
                                             "No", "Yes"))

# Double checking the target variable 'Churn' is a factor
if(!is.factor(churn_data$Churn)) {
  churn_data$Churn <- as.factor(churn_data$Churn)
}
# The ! is the 'not equal to' aka 'not' sign
# What the simple looking if function does is, 
# if the Churn variable is not a factor, it converts it into a factor.
# Very simple yet effective, because we deploy the if function here.
# ------------------------------------------------------------------------------
# ACTUARIAL APPROACH - CATEGORISING BASED ON RISK
# ------------------------------------------------------------------------------

# We know that risk is rarely linear. 
# A customer with 2 months tenure is very different from one with 70 months.
# We will create a discrete 'TenureGroup' variable to capture cohorts.
# This is done just to split the risk categories to study them better in the 
# EDA section. This does not change the dataset in any form apart from adding
# this one additional column/covariate.

# Checking range of tenure first
range(churn_data$tenure) # Tenure ranges from 0 months to 72 months
# Seems like sensible values indeed

# Creating the cohorts
# Group 1: 0-12 months (New customers, possibly high risk - will explore later)
# Group 2: 12-24 months
# Group 3: 24-48 months
# Group 4: 48-60 months
# Group 5: > 60 months (Loyal customers, possibly low risk - will explore later)
churn_data$TenureGroup <- cut(churn_data$tenure, 
                              breaks = c(-1, 12, 24, 48, 60, 72), 
                              labels = c("0-1 Year", "1-2 Years", "2-4 Years",
                                         "4-5 Years", "5+ Years"))

# Checking the distribution of the new variable
table(churn_data$TenureGroup)
# We have successfully created a new covariate here, with a very good reason.

# ------------------------------------------------------------------------------
# EXPLORATORY DATA ANALYSIS (EDA) - Graphs
# ------------------------------------------------------------------------------
# Here we visualise the distributions to understand the risk profile of 
# the portfolio.

# We need to see if the classes are balanced.
# Imbalanced classes can bias the model towards the majority class.

ggplot(churn_data, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  labs(title = "Churn Distribution", 
       x = "Churn Status", 
       y = "Count") +
  theme_bw() # Keeping themes simple/standard
# This shows that the classes are indeed not 'balanced' per se.
# Most of the customers' status is 'No', so the status with 'Yes' are a 
# minority in this dataset.
# The event being tested for (churn) is rarer than the non-event (retention).

#This suggests that standard accuracy metrics may be misleading as even a 
# naive classification model predicting "No" for everyone would still 
# achieve a high accuracy score (of roughly 73.46%).
# Which means, simple accuracy analyses is not sufficient here.

# This is why, in the future, we will use AUC (Area Under the Curve) 
# and Sensitivity/Specificity as primary performance metrics rather than simple 
# accuracy percentage comparison. 

# Calculating exact proportions for the report
prop.table(table(churn_data$Churn))
# Around 73.46% did not churn, while 26.54% approx. did churn.


# Let us now understand how long customers tend to stay in general.

ggplot(churn_data, aes(x = tenure)) +
  geom_histogram(binwidth=10, fill = "steelblue1", color = "red") +
  labs(title = "Tenure Distribution", x = "Tenure (in Months)",
    y = "Frequency/Count") +theme_minimal()

# We can see that the customers mostly churn within the first few months.
# If they don't churn in the first few months, we can observe that the
# probability of churning has gone down significantly.

# This is not normally distributed, so we would be better off splitting this 
# covariate into categories, such as under six months, 6-12 months etc., which
# is what we had done earlier.



# Let us not plot and study the monthly charges

ggplot(churn_data, aes(x = MonthlyCharges)) +
  geom_histogram(binwidth = 10, fill = "turquoise1", color = "red") +
  labs(title = "Monthly Charges' Distribution",x = "Monthly Charges",
       y = "Frequency")+theme_bw()

# As we can observe, the highest 'charges' is for the initial period, and the 
# amount is rather low (circa £20), indicating the 'basic' package.

# The remaining distribution is more or less as expected,
# but we also observe that the monthly charges above say, £120 ish are very low.
# This shows us that most of the customers are unwilling to spend ostentatious
# amounts of money, but most of them seem to be fine spending upto say, £90
# or so.


# Let us now study the skewness in total revenue collected.
ggplot(churn_data, aes(x = TotalCharges)) +
  geom_histogram( fill = "slategray1", color = "red") +
  labs(title = "Total Revenue distribution",
  x = "Total Revenue",y = "Frequency") +  theme_bw()

# The revenue collected is very heavily positively skewed. This almost
# resembles a 'decay' equation that we model in actuarial cycles.

# This is because total revenue = monthly charges * tenure
# The earlier analyses did indeed show us that most of the churns occur
# very early on, causing the tenure for the same to be very small, causing
# the most frequent total charges to be a very low amount.
# The decay seems to be exponential with a negative coefficient.



# Categorical Variable Analysis
# We look at key demographic and account indicators.

#  Gender
# Does gender impact the probability of churning?
ggplot(churn_data, aes(x = gender, fill = gender)) +
  geom_bar() +labs(title = "Count of Customers by Gender", x = "Gender", 
                   y = "Count") +  theme_bw()

# Gender does not seem to be all that significant. Almost as many females
# churn as males do. However, we do indeed note that slightly more males
# churn compared to females, but it does not seem to be significant enough
# at an initial glance.


#Senior Citizen Status
# Who churn more, younger people or older people? We expect younger people to 
# churn more, as this is consistent with marketing, psychology etc.

ggplot(churn_data, aes(x = SeniorCitizen, fill = SeniorCitizen)) +
  geom_bar() +labs(title = "Count of Senior Citizens", 
                   x = "Is Senior Citizen?", y = "Count") +  theme_bw()

# As we can see, the churn count for senior citizens is indeed extremely low.
# This confirms our earlier suspicions - Senior citizens are in general much
# less tech savvy, and may be more against changes, and hence
# they may choose to continue with the same provider than churn, because
# senior citizens in general prefer predictability and uniformity.



# Partner Status - Does having a partner affect the probability of churning?
ggplot(churn_data, aes(x = Partner, fill = Partner)) +
  geom_bar() +labs(title = "Customers with Partners", 
                   x = "Has Partner?", y = "Count") +  theme_bw()

# We can infer that the partner status does seem to have a significant 
# effect on churn status. It is likely that those without partners do
# churn slightly more than those who do, possibly owing to the former group
# not valuing stability and consistency as much.



# Dependents Status - Having a dependent or more, will it affect churn status?
# Logically, one could preconceivedly argue that those with dependents are in
# general more resistant to changes, so we would hypothesise that the ones 
# without dependents may be more likely to churn.

ggplot(churn_data, aes(x = Dependents, fill = Dependents)) +
  geom_bar() +labs(title = "Customers with Dependents", 
                   x = "Has Dependents?", y = "Count") +  theme_bw()

# We can observe a rather strong relationship here. Those without dependents
# are much likely to churn compared to those who do not. This relationship
# is indeed quite significant.

# This is usually because those without dependents are usually younger people,
# who tend to experiment more in life, in general. They tend to also be more
# informed, more tech savvy and are not averse to change in general, as they
# have grown up in our highly changing world, and are no strangers to change.


# Bivariate Analysis (Key Predictors vs Churn)

# We want to visually inspect which variables separate "Yes" from "No".

# Churn by Contract Type
# Hypothesis: Month-to-month contracts are riskier, because the ones taking up
# this kind of a contract may be more susceptible to churning. The implication
# of this hypothesis extends to a fact that long term customers are less 
# risky and are less likely to churn.

ggplot(churn_data, aes(x = Contract, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Contract Type", 
       x = "Contract Type", 
       y = "Proportion") +
  scale_y_continuous(labels = scales::percent) + 
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )
# Note: Position="fill" helps us see the ratio/probability, 
# which is useful for risk.

# Our initial assumption/hypothesis has been proven correct. Month-to-month
# contract is indeed the one that is most susceptible to churn risk. We also 
# observe that as the contract length increases, the probability of churning 
# decreases. 

# Churn by Payment Method
# Hypothesis: Electronic check users might churn more due to ease of switching.
# Implication - those paying using card etc. are expected to be less likely
# to churn.

ggplot(churn_data, aes(x = PaymentMethod, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn and Method of Paying", 
 x = "Payment Method", y = "Proportion/Percentage") +
 scale_y_continuous(labels = scales::percent) +theme_bw() +
 theme(axis.text.x = element_text(angle = 72, hjust = 1)) 
# Rotating the labels for readability and 'offsetting' the height aka
# adjusting the height by one so that the x axis labels do not overlap with 
# the graph itself

# Our initial hypothesis has indeed been proven correct. The ones making
# electronic payments are indeed significantly more likely to churn compared
# to those who make payments via other methods.
# One most likely reason is that those making electronic payments happen to be
# facing less inertia, and may thence be more open to changes.

# The other forms of payments (especially mailed check, etc.) are generally made
# by those who are less tech-savvy and/or much older, so they are not so
# explorative, and prefer to stick with what they have rather than undergoing
# changes.

# The credit card payments/automatic ones are usually the least hassle free,
# and psychologically, the customers under those plans may not feel compelled
# to change at all due to how at-ease they feel with their current system.



# Churn by Internet Service type
# Fiber optic is more expensive, so, does it drive more churn?
ggplot(churn_data, aes(x = InternetService, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Internet Service Type", 
  x = "Internet Service", y = "Proportion") +
  scale_y_continuous(labels = scales::percent) +  theme_bw()

# Yes, our initial hypothesis has been proven right once again. Fiber optic,
# being the most expensive, does indeed have significantly higher churn rates
# than the other internet services.

# This again likely ties back to age and or how informed the customers are.
# Fiber optics are usually the most cutting edge technology, and more tech-savvy
# and/or younger people usually prefer it. Since the most tech-savvy market
# is always very competitive, it may be that the current provider may not be 
# offering the best value for money at times, driving the churn rates.


# Churn by Tenure Group (Our custom-created covariate for better understanding)
# We expect newer customers to be much higher risk and established ones to be 
# of lower risk.
ggplot(churn_data, aes(x = TenureGroup, fill = Churn)) +
  geom_bar(position = "fill") +
  labs(title = "Churn Rate by Tenure Cohort",x = "Tenure Group",
y = "Proportion") +  scale_y_continuous(labels = scales::percent) +theme_bw()

# Our suspicions are confirmed once again. The customers who are in the 0-1
# year tenure are the most likely group to churn. We also observe another
# important trend/feature. Which is - As the tenure increases, the probability
# of churning keeps decreasing, which is in line with what we would expect from
# a psychology/marketing point of view.


# Monthly Charges vs Churn (Boxplot)
# Do people who pay more churn more?
 ggplot(churn_data, aes(x = Churn, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Monthly Charges distribution by Churn Status", 
  x = "Churn", y = "Monthly Charges") +theme_bw()
# Yes, our suspicions are confirmed once again. Those that pay more are indeed
# more likely to churn. The average price paid by those who churn is also
# significantly higher than that for those who do not churn.

# This is likely because those that pay higher prices access more advanced
# features and services, and as a result, they may expect cutting-edge and
# highly competitive technologies. If a competitor, for instance, makes a
# breakthrough in terms of offering a new technology, the members in this group
# (the group being the ones who pay the highest amount) will be willing to churn
 
# Those who are after top-tier services and cutting-edge offerings are seldom
# loyal to a particular brand.


# ------------------------------------------------------------------------------
# CORRELATION ANALYSIS (NUMERIC PREDICTORS)
# ------------------------------------------------------------------------------
# Checking for multi-collinearity among numeric variables.
# We must select only numeric columns for the correlation matrix.

numeric_vars <- sapply(churn_data, is.numeric) # Only for the numeric covariates
correlation_matrix <- cor(churn_data[, numeric_vars])

# Printing the correlation matrix
correlation_matrix

# Visualising the correlation
# This helps identify if variables like TotalCharges and Tenure are highly
# correlated.They usually are.
corrplot(correlation_matrix, method = "number", type = "upper", tl.cex = 0.8)
# Observe how correlated TotalCharges and Tenure are
# As we had discussed earlier, the total charges are a product of tenure and
# monthly charges.

# Hence the reason why Tenure and monthly charges are highly correlated too.
# ------------------------------------------------------------------------------
# HYPOTHESIS TESTING & CONFIDENCE INTERVALS
# ------------------------------------------------------------------------------

# Chi-Square Goodness of Fit Tests
# ------------------------------------------------------------------------------

# Churn and Gender
# Does Gender have any significant relationship with Churn?
# The null hypothesis states that there is no relationship between the two 
# covariates under consideration, the alternative is that there is definitely
# a statistically significant relationship between the two.

table(churn_data$Churn, churn_data$gender)
chisq.test(table(churn_data$Churn, churn_data$gender))
# The p-val of 0.4866 means that we cannot reject the null hypothesis.
# This indicates that gender does not seem to be a significant covariate. 
# This is expected because our plots earlier almost confirmed the same too. 

# Churn and Partner Status
# Does having a partner affect the churn probability?
table(churn_data$Churn, churn_data$Partner)
chisq.test(table(churn_data$Churn, churn_data$Partner))
# The p-value is extremely small.
# Hence, the null hypothesis which states that there is no relationship between
# the two covariates is rejected.
# Not having a partner does indeed increase the probability of churning.


# Churn and Contract
# Checking dependency between Contract type and Churn status
# The null hypothesis states that there is no relationship between the two 
# covariates under consideration, the alternative is that there is definitely
# a statistically significant relationship between the two.

table(churn_data$Churn, churn_data$Contract)
chisq.test(table(churn_data$Churn, churn_data$Contract))

# The p-value is practically zero, which means that we have very strong
# evidence to reject the null hypothesis, even at the 0.00001% significance 
# level, let us say.
# The p-value for this test is practically zero which suggests that
# the relationship is indeed highly significant.
# That is, the contract type and churn have extremely strong relationship.
# This is expected from what we found out earlier.



# Confidence Intervals
# ------------------------------------------------------------------------------

# Difference in Mean Tenure
t.test(tenure ~ Churn, data = churn_data)
# The null hypothesis here is that both those who churn and those who do not
# churn have the same mean tenure. The alternative hypothesis is two sided,
# which means that the alternative hypothesis states that the mean difference
# between the two is NOT zero.


# The p-value for this test is practically zero. This means that we must reject
# the null hypothesis which states that both those who churn and those who don't
# have the same mean tenures.

# Practically, as we had seen earlier, most of those who churned did so during
# the earlier periods of the contract. As a result, we can safely say that those
# who churned have a much, much lower average tenure compared to those who did
# not.

# The mean tenure for the non-churners is ~37+ months
# The mean tenure for the churners is only around 18 months

# Proportions Test to get the 95% CI for the overall Churn Rate
# That is, what is the 95% confidence interval for the churn rate.
prop.test(sum(churn_data$Churn == "Yes"), nrow(churn_data))

# The 95% confidence interval for the churn rate is (25.51%, 27.59%)
# This interval is not too large and is quite convenient and indicates that
# the model is indeed quite reliable.
# This means that even a naive classification model is likely to achieve 
# accuracy scores of >73% or so. In order to get a model that has significant
# predictive powers over the naive model, its accuracy should at least surpass
# 73% or so.



# End of EDA Phase
################################################################################
################################################################################


# ==============================================================================
# SECTION 2: MODEL FITTING AND ANALYSIS - Logistic Regression Models
# ==============================================================================
# ------------------------------------------------------------------------------
# LOGISTIC REGRESSION (GLM) - The Baseline for all classification problems
# ------------------------------------------------------------------------------

# Firstly, let us split the data into Training and Testing sets
# We use a 70-30 approach as is the convention in machine learning
# That is, 70% of the data will be used for training, 
# and the remaining 30% will be independently used for testing our model.

set.seed(10) # Setting seed again just to be safe
# We had already set the seed at the start, but we do it again nonetheless.
train_index <- createDataPartition(churn_data$Churn, p = 0.70, list = FALSE)

train_data <- churn_data[train_index, ]
test_data  <- churn_data[-train_index, ]

# Common sense Check: Verifying class distribution in the splits
# It is important that the proportions are similar in both sets.

# Class distribution in Training Set
prop.table(table(train_data$Churn))
# As we can see, approx 73.43% are No, 26.54% are Yes
# These probabilities are very close to the probabilities of the actual data,
# so the training data gets the approval.

prop.table(table(test_data$Churn))
# Again, the probabilities are indeed very very close to the ones for the 
# training data, and by extension, to the actual full dataset too.
# This is very satisfactory, so the test data split get the approval too.

# We did the above check because for some seeds, there is a chance, albeit
# very very low, that the splits may be completely disproportionate to the
# actual probabilities/splits, which renders the training and testing 
# inaccurate, hence, we just checked to the see the proportions for those too.

# Fortunately, these are satisfactory indeed, and the proportions are quite 
# similar to what is observed in the actual data.


# We use the binomial family for binary classification (Yes/No).
# Recall that this is a 'classification' problem, so this is the approach taken.


# Let us now fit the logistic regression model
# Fitting the full model first to see significance of all variables
glm_full <- glm(Churn ~ ., data = train_data, family = "binomial")
# We have fit the model on the training data split, because we will firstly
# train the model on this dataset, and then we will test it on unseen data
# which will come from the testing data split.

#Let us see the summary of the fitted model to study and understand it.

summary(glm_full)
# Observe that the p-value for PartnerYes is 0.999770, which is extremely
# high indicating that the partner variable is practically useless.
# There are some other covariates in there that are insignificant too.
# For instance, tech support has a p-value of 0.8408, genderMale has a p-value
# of 0.688
# However, note that some of these are classic cases of multi-collinearity.
# E.g., if InternetService is No, then all internet related columns appear
# insignificant. 

# The key variables seem to be the ones who pay using electronic check (manual)
# the ones who have InternetService as Fiber.
# Similarly, the low tenure contracts seem to be having a significant effect.

# We do not remove any covariates immediately, let us first run AIC/BIC
# before deciding which covariates to remove. This gives us formal conviction
# and statistical/mathematical reasons for us to remove insignificant covariates

# ------------------------------------------------------------------------------
# GLM Prediction
# ------------------------------------------------------------------------------
# Predicting probabilities on the testing split
glm_probs <- predict(glm_full, newdata = test_data, type = "response")

glm_pred <- ifelse(glm_probs > 0.5, "Yes", "No")
# Convert probabilities to class labels (using threshold 0.5)
glm_pred <- factor(glm_pred, levels = c("No", "Yes"))


# Confusion Matrix for GLM
confusionMatrix(glm_pred, test_data$Churn, positive = "Yes")

# Note the following. Accuracy is around 80%, which is quite good.
# Sensitivity and specificity have reasonable values of 0.5143 and 0.9059.

# The sensitivity of 51% is not too good. This means that the true positive
# rate is only 51% approx. Here, positive means 'Yes'. That means, of all the
# total positives, the model is capable of identifying 51% of them all.
# That is, the true positive rate is 51%.

# The model may contain redundant
# covariates, which we will remove in the AIC/BIC section and will compare.

# The specificity of 0.9059, in particular, is an excellent number.
# This means that the model has a true negative rate of 90%+. This means that
# the model is capable of identifying over 90% of all true negative cases.
# This is an excellent number as far as the specificity is concerned.

# P-Value [Acc > NIR] : 3.267e-13 - This means that our model does indeed
# possess predictive power, and is better than a naive classification approach.
# Here, the null hypothesis was that our model is not better than the naive
# classification approach, which is rejected due to the very low p-value.

###############################################################################

# ------------------------------------------------------------------------------
# Model Selection using stepAIC - Akaike Information Criterion
# ------------------------------------------------------------------------------
# We use stepAIC to perform stepwise selection. 
# AIC balances the goodness of fit against the complexity of the model.
# It penalises the model for complexity, but also rewards it for accuracy.
# This enables us to get a parsimonious model.

glm_aic <- stepAIC(glm_full, direction = "both", trace = 0)
# Direction = both is crucial for AIC, because it will consider removing 
# covariates and adding covariates at each stage with the ultimate aim
# of coming up with a model that has the lowest AIC among all the other
# models considered.


# Display the summary of the AIC-optimised model
summary(glm_aic)

# ------------------------------------------------------------------------------
# Model Selection using stepAIC (BIC) - Bayes Information Criterion
# ------------------------------------------------------------------------------
# By setting k = log(n), stepAIC calculates BIC. 
# BIC is more 'strict' and often leads to a simpler model (parsimony).
# AIC and BIC are very similar, but differ in how the penalty term operates.

n_obs <- nrow(churn_data)
glm_bic <- stepAIC(glm_full, direction = "both", k = log(n_obs), trace = 0)

# Displaying the summary of the BIC-optimised model
summary(glm_bic)

# ------------------------------------------------------------------------------
# Comparison of AIC vs BIC
# ------------------------------------------------------------------------------
# Typically, the BIC model will have fewer variables. 
# This is because the penalty the BIC imposes is 'harsher' and leads to stricter
# conditions.

# We notice that the AIC retains Dependents, DeviceProtection, and,very 
# crucially, both Tenure and TenureGroup (both are the same, but in
# different formats) - Recall that TenureGroup is our custom covariate

# The BIC, to contrast, dropped Dependents, DeviceProtection and TenureGroup
# but retained the Tenure covariate.

# The issue with the TenureGroup and Tenure covariate arises from 
# multi-collinearity. The BIC, by dropping TenureGroup, has resolved the issue
# that arose from multi-collinearity.

# Recall that the TenureGroup variable was one that we'd created earlier for
# understanding the risk categories better, and is not native to the dataset.

# The BIC model, with its stricter penalty (log(n)), correctly 
# dropped 'TenureGroup', 'Dependents', and 'DeviceProtection'.

# We can see that the AIC seems to be a bit too 'greedy' in comparison to the 
# BIC. Hence, we choose the resultant model from the BIC as our parsimonious one

# However, there is one tiny issue. We observe NA for StreamingTVNo internet 
# service etc. This is because InternetServiceNo is already in the model.
# If internet service is no, then obviously, streaming service is also no.
# These issues are known as singularities, and have to be dealt with.

# Fortunately for us, the glm() function handles these automatically by ignoring
# them, so we do not need to work on these, thankfully!

# ------------------------------------------------------------------------------
# Final Parsimonious GLM Selection
# ------------------------------------------------------------------------------

# Which model do we select as the parsimonious model?
# The BIC model represents the most parsimonious balance of fit and simplicity.
# We accept the BIC model as our Final GLM.

parsimonious_glm <- glm_bic
# Final check of the coefficients
summary(parsimonious_glm)

# Let us briefly list out the odds ratio here and what they represent
glm_odds_ratio_table <- exp(cbind(OR = coef(parsimonious_glm), 
                          confint(parsimonious_glm)))
glm_odds_ratio_table

# ------------------------------------------------------------------------------
# Odds Ratio Interpretation - For the Parsimonious Model
# ------------------------------------------------------------------------------

# If Odds Ratio is more than 1, this means these covariates increase the
# risk of a customer churning.

# If odds ratio are between 0 and 1, this means that these covariates decrease
# the risk of churning, and are otherwise called protective covariates.

# To get the percentage risk reduction of a protective covariate, 
# we do 1-OddsRatio

# Fiber Optic: The most significant covariate. Customers with Fiber are so much 
# more likely to leave.This suggests a possible deficit with the Optic 
# Fiber products that must be looked into further.

# Two-Year Contracts: A strong protective covariate that decreases the 
# probability of churn. The odds ratio of 0.20 mean these customers are 80% 
# less likely to churn compared to month-to-month users. ( 1 - 0.8 = 0.2 )

# Streaming Movies/TV: Both have odds-ratios of circa 2. Users of these 
# services are twice as likely to churn, likely due to higher costs 
# or lack of novelty in this offering by the firm.

# Tenure: Odds Ratio of 0.93. For every extra month a customer stays, 
# their risk of leaving drops by 7%. 
# This showcases the importance for TelCo to prioritise customer retention.

# Electronic Check: Ratio of 1.5. Customers using this manual payment method 
# are more likely to churn than those on other plans, which we had also 
# identified using our graphs in the EDA section too.

# ==============================================================================
# EVALUATING AND OPTIMISING THE PARSIMONIOUS GLM (BIC MODEL)
# ==============================================================================
# We now evaluate the performance of our chosen parsimonious model (BIC).
# We also explore tuning the classification probability, as the default 0.5 
# cutoff is rarely optimal for imbalanced classification problems.

# Firstly, Standard ROC and AUC Analysis using a threshold of p=0.5
# ------------------------------------------------------------------------------
# Predict probabilities on the held-out Test Data using the BIC model
set.seed(10)
glm_probs_parsimonious <- predict(parsimonious_glm, 
newdata = test_data, type = "response")

# Create the ROC object for the same
roc_glm <- roc(test_data$Churn, glm_probs_parsimonious)

# Predicting probabilities on the testing split
# Convert probabilities to class labels (using threshold 0.5)
glm_pred_parsimonious <- ifelse(glm_probs_parsimonious  > 0.5, "Yes", "No")
glm_pred_parsimonious <- as.factor(glm_pred_parsimonious)

# What we have done here is, we have trained the model using the training data
# (which is 70% of the total data), and then we have used it to predict the
# probabilities for the testing data (30% of the total data), to see how 
# our model actually performs on unseen data.

# Confusion Matrix for GLM
confusionMatrix(glm_pred_parsimonious, test_data$Churn, positive = "Yes")
# Even though we've fit this model after removal of the insignificant covariates
# from BIC, we see that the sensitivity and specificity numbers remain the same
# as above.

# That is, the sensitivity is 54.11%, which means that the model is capable
# of identifying around 54.11% of all positives.
# This is not exactly ideal because the true positive rate being 54.11%
# is very low, and this may lead to underestimation of liabilities.

# The specificity is 90.08%, which means that the model is capable of 
# identifying over 90% of all negative cases. This is an excellent number.

# Accuracy is 80.54%, meaning that the model is indeed better than a naive
# classification model

# P-Value [Acc > NIR] : 2.048e-14, so reject the null hypothesis which states
# that the model is equivalent to the naive model. Hence, we see that p=0.5
# is indeed a model with better predictive powers than the naive model.


# Plot the ROC Curve now to understand how good of a fit the model is.
plot(roc_glm, main = "ROC Curve",col = "steelblue", print.auc = TRUE)

# Print the Area Under the Curve (AUC) - The area under the curve is a measure
# of how good our model is. Generally, the higher the AUC is, the better 
# predictive powers our model will be in possession of.
auc(roc_glm)

# The area under the curve is 0.8453, which is very satisfactory indeed.

# However, p=0.5 seems to be underestimating the liabilities. This is because
# the model seems to account only for 54.11% of the true churners.

# As a result, this may lead to the firm underestimating its liabilities,
# which may, as an extension, cause financial issues, insolvency and ruin risks.


# We always strive for improvement, so let us try a different value of p to see
# how it affects our model's predictions and AUC. We do this mainly because the
# p=0.5 model has a very low sensitivity.


# ------------------------------------------------------------------------------
# Threshold Sensitivity Test (p = 0.8)
# ------------------------------------------------------------------------------
# Perhaps our p=0.5 might be a bit too high.
# Let us try to be very conservative.
# Let's see what happens if we only predict 'Yes' if probability > 0.8.

pred_08 <- ifelse(glm_probs_parsimonious > 0.8, "Yes", "No")
pred_08 <- factor(pred_08, levels = c("No", "Yes"))

# Confusion Matrix
confusionMatrix(pred_08, test_data$Churn, positive = "Yes")

# The sensitivity is 0.3571%, which is abysmally poor to say the least!
# This means that the true positive rate is almost 0, that is, the model never
# seems to give a true positive prediction. This is very disappointing.
# This means that the model will always underestimate the liabilities, which is
# one of the worst effects for a company.
# This is because the model identifies less than 1% of churners, hence,
# the model almost completely fails to acknowledge that churn risk even exists.

# The specificity is almost 100%, meaning that the model's true negative rate
# is essentially close to 100%. This is an excellent number.

# However, the sensitivity rate of 0.3571% is beyond poor, so in reality,
# this model with p=0.8 is disregarded completely without any second thoughts.

# Nonetheless, just to maintain uniformity, let us evaluate its AUC and plot
# the ROC curve.

# ROC and AUC for this threshold (using the binary predictions)
roc_08 <- roc(test_data$Churn, as.numeric(pred_08))
plot(roc_08, main = "ROC Curve with prob 0.8", col = "mediumblue"
     , print.auc = TRUE)
auc(roc_08)

# AUC at p=0.8 is 0.501 approx
# Which is a significant downgrade from our earlier p=0.5, which gave a 
# AUC of over 0.84. This means that increasing the probability seems to yield
# worse results, so we disregard this .

# P-Value [Acc > NIR] : 0.531 
# This p-value suggests that we cannot reject the null hypothesis which states
# that our model has no more predictive powers than the naive model.
# Accuracy : 0.7344
# No Information Rate : 0.7348


# This means that this model for p=0.8 is not any better than a naive 
# classification method, hence, we disregard p=0.8 totally.


# This implies that moving above p=0.5 yields inferior results
# So how about we try and go below p=0.5?

# ------------------------------------------------------------------------------
# Threshold Sensitivity Test (p = 0.4)
# ------------------------------------------------------------------------------
# Try p=0.4 just for the sake of it

pred_04 <- ifelse(glm_probs_parsimonious < 0.4, "No", "Yes")
pred_04 <- factor(pred_04, levels = c("No", "Yes"))

# Confusion Matrix
confusionMatrix(pred_04, test_data$Churn, positive = "Yes")

# The sensitivity is 64%, which is reasonably good. This corresponds
# to the true positive rate.This means that the model is capable of identifying
# upto 64% of all positive cases. While this isn't exactly perfect, we see that
# this is a minor improvement over p=0.5 in terms of not underestimating 
# liabilities.

# Specificity is almost 84%, meaning that the model has a true negative rate
# of 84%, that is, it can identify 84% of all true negative cases, which is 
# quite good.


# P-Value [Acc > NIR] : 4.059e-08, this means reject the null hypothesis 
# which states that the model is equivalent to the naive model. 
# Hence, we see that p=0.4 is indeed a model with better predictive powers than 
# the naive model.

# ROC and AUC for this threshold (using the binary predictions)
roc_04 <- roc(test_data$Churn, as.numeric(pred_04))
plot(roc_04, main = "ROC Curve", col = "hotpink1", print.auc= TRUE)

auc(roc_04)
# Area under the curve is 0.7411, which is poorer than what we had for p=0.5
# However, it is definitely an improvement when it comes to the balance between
# sensitivity and specificity, so for practical purposes, even if it may have
# lower AUC, a balance of sensitivity and specificity is preferred in practice,
# so in many cases, this probability may be chosen over p=0.5 model.

# We have somewhat discerned that lowering the probability from 0.5 seems to
# make the sensitivity and specificity more balanced, but which is the optimal
# point that balances both?


# We have done trial and error thus so far to observe the 'trend'.
# This is how we discerned that increasing the probability greatly ruins the
# balance between sensitivity and specificity.

# Rather than randomly performing trial and error, let us do this in a 
# rather systematic way.
# ------------------------------------------------------------------------------
#  Optimising Threshold using Youden's J Statistic
# ------------------------------------------------------------------------------
# Youden's J is equal to (Sensitivity + Specificity - 1)
# While there is no one single point that may maximise AUC per se, 
# a good point to consider is the Youden's J statistic.
# This statistic finds the optimal point that maximises the
# difference between Sensitivity and Specificity.
# In other words, it is the most optimal way to balance sensitivity and
# specificity.

# The issue we had faced above is that when sensitivity improves, specificity
# degrades, and the vice-versa is true too. The Youden's statistic gives the
# probability that has the optimal 'balance' between sensitivity and 
# specificity.
set.seed(10) # Setting the seed again just to be sure

# Set the coordinates firstly to obtain Youden's J
optimum_coordinates <- coords(roc_glm, "best", best.method = "youden",
                              levels = c("No", "Yes"), direction = "<")
optimum_coordinates
#  threshold specificity sensitivity
#  0.2429901   0.7094072   0.8232143
# We have here obtained the threshold, which is another name for the 
# classification probability that maximises Youden's J Statistic.

# Hence, the Youden's J Statistic here is p=0.2429901   

# Extract the specific threshold value
youden_threshold <- optimum_coordinates$threshold
youden_threshold #p=0.2429901 seems to be the best threshold

# Round it to 3 decimal places
youden_threshold <- round(youden_threshold,3)

# Apply the optimal threshold to create class predictions
# If probability < youden_threshold, classify as No, else Yes
# This is simple binary classification, as we have always done
prediction_Youden <- ifelse(glm_probs < youden_threshold, "No", "Yes")
prediction_Youden <- factor(prediction_Youden, levels = c("No", "Yes"))

# Generate the Confusion Matrix
# Generating the confusionMatrix for prob equals Youden's J Statistic
confusionMatrix(prediction_Youden, test_data$Churn, positive = "Yes")
# As usual, we are generating the confusion matrix here, and we say that
# the positive case occurs when churn = 'Yes', just to be safe.

# The sensitivity for this model is 82.3% approx, which means that the model
# has a true positive rate of 82.3%. That is, of all the positives cases that 
# exist, the model is capable of identifying a respectable 82%+ of them.
# This is a satisfactory sensitivity rate. This corresponds to the true 
# positive rate.

# The specificity of this model is around 71.26%, which means that the model is
# capable of identifying 71.26% of all negative cases. This corresponds to the 
# true negative rate. This is quite pleasing, especially since the model has a 
# very good sensitivity rate too (true positive).

# P-Value [Acc > NIR] : 0.238
# The null hypothesis for the above test states that the model under 
# consideration is not significantly better than a naive classification model.
# Even though this figure may seem misleading, it reflects the intentional 
# trade-off required to maximise sensitivity because for our case, financial
# liabilities are closely tied to the model sensitivity and an overall
# high accuracy score is only secondary.

# These are indeed satisfactory compared to p=0.5 case, which had a very high
# specificity (true negative rate), but a non-satisfactory sensitivity 
# (corresponding to the true positive rate)

# In any case, let us work out the AUC and plot the ROC curve too

# ROC and AUC for the Optimal Threshold Model
# We treat the binary class predictions as numeric to plot the ROC curve
roc_optimal <- roc(test_data$Churn, as.numeric(prediction_Youden))

# Plotting the ROC for this specific threshold
plot(roc_optimal, main = "ROC Curve", col = "lightcoral", print.auc = TRUE)
# The straight line is the line of no discrimination, that is, the line with
# no predictive powers. The lightcoral line is our ROC curve. The area between
# the straight line and our ROC curve's line is an indicator of how good
# our model is. Here, is it satisfactorily good.

# Print the final AUC for this threshold
auc(roc_optimal) # 0.768
# ------------------------------------------------------------------------------
# Evaluation of our GLM models with different tuned parameters
# ------------------------------------------------------------------------------

# The main question we need to ask ourselves now - Is this an improvement?
# Our model with p=0.5 had 51.4% sensitivity, and 90.6% specificity.
# Our model with Youden's statistic has 82.3% sensitivity, and 71.3% specificity

# p=0.5 had a low sensitivity rate, which was quite bad. When p is tuned to 
# Youden's J, we are catching 82%+ of all positives as true positives.

# Although this lowers the overall AUC slightly (drop from 0.8453 to
# 0.768) to do so. We have sacrificed some specificity (true negative rate) to
# obtain this balance.


# The model with Youden's J Statistic essentially treats both classes in a more
# balanced way, because the true positive rate is not too far off from true 
# negative rate. This is precisely what we want, because this seems to be 
# counteracting the issues that arise due to the class imbalance we had pointed 
# out earlier.

# Hence, in an ideal scenario, rather than using p=0.5 or p=0.4, we would select
# p=0.2429901, optimised using Youden's J, for the logistic regression 
# model from a practical standpoint.


# ------------------------------------------------------------------------------
# END OF GLM OPTIMISATION
# ------------------------------------------------------------------------------
# We now proceed to Regularisation (Lasso/Ridge) to see if we can 
# improve upon this selection even further.


# ==============================================================================
# SECTION 3: PENALISED REGRESSION (RIDGE & LASSO)
# ==============================================================================
# Penalised regression is used to reduce overfitting and handle 
# multicollinearity.
# Ridge (L2) shrinks coefficients, while Lasso (L1) can perform variable 
# selection.

# Regularisation is primarily used to prevent overfitting, as it has a penalty
# terms that punishes the same. 

# Both models apply a penalty to large coefficients, thereby shrinking them.

# The difference is that Ridge Regression uses a L2 penalty (squared), because 
# of which it may shrink terms close to zero, but terms are seldom reduced 
# to zero.

# On the other hand, Lasso has a L1 penalty in force (Modulus penalty), which
# has the ability to shrink coefficients to exactly zero, and as needed, can
# also be used to perform variable selection.

# The previous models (logistic ones) did not have a penalty for complexity.
# In reality, however, we know that computing power is limited, and programmers
# and analysts are always questing for the holy grail of optimised accuracy.

# These methods also handle large datasets slightly better than the glm()
# function, and hence, we use these here too for our analyses.

# ------------------------------------------------------------------------------
#  Data preparation before using glmnet package's functions
# ------------------------------------------------------------------------------
# glmnet requires covariates to be supplied as a numeric matrix.
# Factors are converted into dummy variables using model.matrix().

glmnet_matrixprep <- model.matrix(Churn ~ ., data = churn_data)[, -1]
# The intercept is always removed - This is a very important step too.
factor_y_glm <- churn_data$Churn # The y-factor is just our Churn classification

# We use the same train-test split as earlier models
x_train <- glmnet_matrixprep[train_index, ]
x_test  <- glmnet_matrixprep[-train_index, ]

train_y_glm <- factor_y_glm[train_index]
y_test  <- factor_y_glm[-train_index]

# Convert response to numeric for glmnet (Yes = 1, No = 0)
train_y_glm_num <- ifelse(train_y_glm == "Yes", 1, 0)
y_test_num  <- ifelse(y_test  == "Yes", 1, 0)

# Factor version retained for confusion matrices
y_test_factor <- factor(y_test, levels = c("No", "Yes"))

# ------------------------------------------------------------------------------
#  Ridge Regression (L2 penalty, alpha = 0)
# ------------------------------------------------------------------------------
# Ridge regression shrinks coefficients towards zero but keeps all predictors.
# It uses a squared error as the loss function, thereby ensuring that
# coefficients shrink adequately but not exactly to zero.

set.seed(10) # I try to almost always use a seed of 10

# Using the cv.glmnet() function to fit a ridge regression
# model using cross validation
ridge_regression <- cv.glmnet(x_train,train_y_glm_num,family = "binomial",
                              alpha = 0,type.measure = "auc")

# Optimal penalty parameter
# Here, we need the parameter that optimises performance, aka minimises errors
# A good estimate of lambda is imperative for regularisation methods to work
# efficiently and produce good results.
ridge_bestlambda_estimate <- ridge_regression$lambda.min

# Predict probabilities on test data
prediction_ridge <- predict(ridge_regression,s = ridge_bestlambda_estimate,
newx = x_test,type = "response")

# What we have done here, as usual, is fit the model on the training data (70%),
# and then we predict the probabilities on the test data (30%).

# Convert probabilities to class labels - Simply binary classification
ridge_classification <- ifelse(prediction_ridge>=0.5, "Yes", "No")
ridge_classification <- factor(ridge_classification, levels = c("No", "Yes"))

# Model performance - As usual, let us get the confusion matrix 
confusionMatrix(ridge_classification, y_test_factor, positive = "Yes")

# Sensitivity is 0.4983, which means that of all the positives, the model is
# capable of identifying only 49.83% of them correctly. This is the true 
# positive rate. This number is not quite satisfactory, but we observe that
# it is quite similar to what we had obtained with logistic regression with 
# p=0.5, and, if anything, is a slightly worse result.

# Specificity is 0.9143, which means that the model is capable of correctly
# identifying over 91% of all true negatives. This is a very satisfactory 
# number as far as the true negative rate is concerned, however, the sensitivity
# is a bit low, comparatively.

# The overall accuracy is around 80.1%. Recall that due to the class imbalance,
# even a naive classification method would yield accuracies of upto 73%. Even
# with that being said, this model definitely is superior to the naive model.

#P-Value [Acc > NIR] : 4.795e-13      
# For the above, the null hypothesis states that the model in consideration
# has no better predictive powers than the naive classification model,
# which is rejected due to the very low p-value. This means that the model
# here is indeed better than the naive classification model.


# Now, it's time to plot the ROC curve.
roc_ridge <- roc(y_test_num, as.numeric(prediction_ridge))
auc(roc_ridge)
# Area under the curve: 0.8469
plot(roc_ridge, main = "ROC Curve",col="seagreen1",print.auc=TRUE)

# The ROC curve's straight line is the line of no discrimination, and here,
# the seagreen coloured curve is the curve associated with Ridge regression.
# The area between the line of no discrimination and the curve in question
# is a representation of how good our method really is.
# Here, the AUC is 0.847, which is quite good, because it tells us that our 
# model does indeed have good predictive powers.


# ------------------------------------------------------------------------------
#  Lasso Regression (L1 penalty, alpha = 1)
# ------------------------------------------------------------------------------
# Lasso regression can set some coefficients exactly to zero, performing
# automatic feature selection.

set.seed(10) # I always use a seed of 10 :)
# Using the cv.glmnet() function to fit a lasso model using cross validation
lasso_regression <- cv.glmnet(x_train,train_y_glm_num,family = "binomial",
                              alpha = 1,type.measure = "auc")


# Here, we need the parameter that optimises performance, aka minimises errors.
# A good estimate of lambda is imperative for regularisation methods to work
# efficiently and produce good results.
lasso_bestlambda_estimate <- lasso_regression$lambda.min

# Inspect the variables
lasso_coefficients <- coef(lasso_regression, s = lasso_bestlambda_estimate)
lasso_coefficients[lasso_coefficients[, 1] != 0, ]

# Predict probabilities on test data
prediction_lasso <- predict(lasso_regression,s = lasso_bestlambda_estimate,
newx = x_test,type = "response")

# Convert probabilities to class labels
lasso_classification <- ifelse(prediction_lasso >= 0.5, "Yes", "No")
lasso_classification <- factor(lasso_classification, levels = c("No", "Yes"))

# Model performance
confusionMatrix(lasso_classification, y_test_factor, positive = "Yes")

# These results are almost identical to what we got from Ridge regression.
# Sensitivity is 0.5089, which means that of all the positives, the model is 
# capable of identifying only around 51% of them correctly. This is not the
# most ideal of results, but this is to be expected with imbalanced classes
# like the one here.

# Specificity is 0.9104, which means that the model is capable of correctly
# identifying around 91% of all negatives correctly. This is another way of 
# saying that the true negative rate is close to 91%. As far as the true 
# negative rates themselves are concerned, this is an excellent number,
# but the low true positive rate (sensitivity) is a bit of a disadvantage here
# too, as was the case in Ridge Regression and Logistic regression with p=0.5

#    P-Value [Acc > NIR] : 3.247e-11      
# For the above, the null hypothesis states that the model in consideration
# has no better predictive powers than the naive classification model,
# which is rejected due to the very low p-value. This means that the model
# here is indeed better than the naive classification model.



roc_lasso <- roc(y_test_num, as.numeric(prediction_lasso))
auc(roc_lasso)
# Area under the curve: 0.8469 - This is a very good AUC score
plot(roc_lasso, main = "ROC Curve",col="deepskyblue",print.auc = TRUE)
# The ROC curve's straight line is the line of no discrimination, and here,
# the deepskyblue coloured curve is the curve associated with Lasso regression.

# The area between the line of no discrimination and the curve in question
# is a representation of how good our method really is.
# Here, the AUC is 0.843, which is quite good, because it tells us that our 
# model does indeed have good predictive powers.

# The AUC of Ridge Regression is 0.847 while the AUC of Lasso regression is
# around 84.7% too. Furthermore, both have almost identitical sensitivities,
# specificities and the same results.

# Due to the slightly better balance of sensitivity and specificity, we could
# argue that the lasso model is slightly better than the ridge regression 
# model, however, the difference is minuscule here.

# Hence, we say that Ridge and LASSO regressions give almost identical results, 
# for all intents and purposes.
# ------------------------------------------------------------------------------
# End of penalised regression models
# ------------------------------------------------------------------------------

# ==============================================================================
# SECTION 4: K-NEAREST NEIGHBOURS (KNN)
# ==============================================================================

# Separate predictors and response
# We know KNN is a distance based measure (Euclidean distance)
# KNN works only on numeric predictors, so the response variable is removed

x_train_knn <- train_data[, setdiff(names(train_data), "Churn")]
x_test_knn  <- test_data[, setdiff(names(test_data), "Churn")]

y_train_knn <- train_data$Churn
y_test_knn  <- test_data$Churn

# Convert categorical variables into dummy variables
# KNN cannot handle factors directly

x_train_knn <- model.matrix(~ . - 1, data = x_train_knn)
x_test_knn  <- model.matrix(~ . - 1, data = x_test_knn)


# ------------------------------------------------------------------------------
# Scaling predictors
# Distance-based methods require standardised variables
# We scale using training data, and test using testing data, as usual.
# Different covariates have different measures in different units and scales
# If we do not standardise, then the one that is the largest will always 
# dominate and will lead to incorrect results.

# As a result, standardising is always mandatory for KNN.
# ------------------------------------------------------------------------------

x_train_scaled <- scale(x_train_knn)

x_test_scaled <- scale(
  x_test_knn,
  center = attr(x_train_scaled, "scaled:center"),
  scale  = attr(x_train_scaled, "scaled:scale")
)
# ------------------------------------------------------------------------------
# Choosing k values
# We use odd values only to avoid ties
# Small k -> high variance
# Large k -> high bias
# As with everything else, the key thing to consider is bias-variance tradeoff
# ------------------------------------------------------------------------------

k_values <- seq(1, 31, by = 2)
# K values here are odd, i.e., 1, 3, 5, 7, 9 etc. all the way until 31

accuracy_knn <- numeric(length(k_values))
# The accuracy_knn value will get filled up inside our for loop below
# For now, we just define it with its expected length to be the same length as
# the number of k values, because each k value will correspond to one unique
# accuracy.

# ------------------------------------------------------------------------------
# Manual KNN parameter tuning using for-loop
# ------------------------------------------------------------------------------

set.seed(10) # Setting the seed for reproducibility

for (i in seq_along(k_values)) { # The loop runs from 1, 3, 5, 7 all the way 
  # until 31, as needed.
  
  knn_pred <- knn(
    train = x_train_scaled,
    test  = x_test_scaled,
    cl    = y_train_knn,
    k     = k_values[i]
  ) # Here, the KNN model is being fit for each k value under consideration
    # inside the for loop
  
  
  accuracy_knn[i] <- mean(knn_pred == y_test_knn)
  # This is the formula we use for calculating KNN's accuracy
  # There is no one single accuracy formula here like there was for some of our
  # previous models, so we take the average of how many times our predictions
  # match the test values.
}

# Store results neatly for good visual display
knn_results <- data.frame(
  k = k_values,
  Accuracy = accuracy_knn
)

knn_results

# Best k based on accuracy
best_k <- knn_results$k[which.max(knn_results$Accuracy)]
# This line of code enables us to pick the value of k for which the accuracy
# is maximum, amongst all possible values of k we had considered earlier.

best_k 
# The best estimate for k seems to be 25
# In the range 1, 3, 5 etc. all the way until 31, for this particular seed, k=25
# appears to balance both bias and variance optimally, hence we pick this for 
# our future calculations later (ConfusionMatrix etc)

# Smaller k values lead to overfitting
# Larger k values lead to higher bias
# In this case, k=25 balances these two out in the most optimal way

# ------------------------------------------------------------------------------
# Final KNN model using optimal k
# ------------------------------------------------------------------------------

knn_class <- knn(train = x_train_scaled,test=x_test_scaled,
                 cl=y_train_knn,k= best_k) 
# This is the model for our optimal k, which we will use later for other 
# purposes



# Confusion Matrix
confusionMatrix(knn_class, y_test_knn, positive = "Yes")

# Let us interpret the results we get from the confusion matrix
# The sensitivity is 55%, which means that identifies 55% of all true positives
# This is not very satisfactory, but this is to be expected as we had similar
# results with the logistic regression model with p=0.5 too

# The specificity is roughly 88%, which means that it identifies 88% of all 
# true negatives in the model. This is a very satisfactory number.

# The accuracy is 0.7902, which is reasonable
# The 95% Confidence interval for the accuracy is (0.7722, 0.8074), which is 
# also a reasonable interval.

# P-Value [Acc > NIR] : 2.023e-09
# The above p-value is for the null hypothesis which states that our model
# is not better than the naive classification model. Since the p-value is very
# low, we reject the null hypothesis and say that this model is indeed better
# than the naive classification model

# KNN prioritises avoiding false positives over catching churners.


# ------------------------------------------------------------------------------
# ROC Curve and AUC
# ------------------------------------------------------------------------------

# Convert classification predictions to numeric for ROC analysis
knn_probs <- ifelse(knn_class == "Yes", 1, 0)

roc_knn <- roc(y_test_knn, knn_probs, print.auc = TRUE)
plot(roc_knn, main="ROC Curve", col="plum1", print.auc=TRUE)
auc(roc_knn)
# Area under the curve: 0.715

# This model derived from the KNN (with k=25, our best estimate) is indeed
# significantly better than the naive classification model, but, it is not 
# strong enough to rely on classifying churn by itself.
# The logistic model with the classification probability threshold derived
# from Youden's J statistic does indeed still seem like the favourite so far.

# Now that we have used traditional statistical models and machine learning 
# models, we have to do a model based on deep learning.


# ==============================================================================
# SECTION 5: Deep Learning Model (Neural Network)
# ==============================================================================

# Generally speaking, deep learning models use neural networks that mimic the
# functioning of a human brain and can see and decode relationships that are
# not immediately visible on the outside.

# ------------------------------------------------------------------------------
# Data Preparation
# ------------------------------------------------------------------------------
# Neural networks are sensitive to scale, so we standardise numeric predictors.
# We reuse the same train-test split from earlier for re-usability.

# Extraction of predictors and response variable
x_neural_network <- churn_data[, setdiff(names(churn_data), "Churn")]
y_neural_network <- churn_data$Churn

# Conversion of factors to dummy variables
x_neural_network_mat <- model.matrix(~ . -1, data = x_neural_network)

# Train-test split
x_train_neural_network <- x_neural_network_mat[train_index, ]
x_test_neural_network  <- x_neural_network_mat[-train_index, ]

y_train_neural_network <- y_neural_network[train_index]
y_test_neural_network  <- y_neural_network[-train_index]

# Scaling of predictors using training data
x_train_neural_network_scaled <- scale(x_train_neural_network)
x_test_neural_network_scaled  <- scale(
  x_test_neural_network,
  center = attr(x_train_neural_network_scaled, "scaled:center"),
  scale  = attr(x_train_neural_network_scaled, "scaled:scale")
)

# As was the case for KNNs, the predictor variables must be scaled for neural
# networks/deep learning too. This is because if we do not do this, the 
# covariate with the highest units/dimensions will dominate our calculations
# and would lead to incorrect results.

# ------------------------------------------------------------------------------
#  Model Fitting
# ------------------------------------------------------------------------------
# We fit a small neural network with:
# one hidden layer
# a reasonable number of neurons
# weight decay to prevent overfitting

set.seed(10)

fit_neuralnetwork_model <- nnet(
  x = x_train_neural_network_scaled,
  y = class.ind(y_train_neural_network),
  size = 5,        # number of hidden neurons
  decay = 0.01,    # regularisation parameter
  maxit = 200,     # maximum iterations
  softmax = TRUE, # multi-class probability output
  trace = FALSE
)
# The decay parameter is the weight decay, which is a regularisation technique
# that penalises large weights. This is done with the intention of preventing
# overfitting. Philosophically, this is not too different from Lasso/Ridge 
# regression models.

# The softmax=true converts logistic values into probabilities, ensuring that
# all the outputs are between 0 and 1, and that all probabilities together
# sum up to one.

# Maxit=200 sets the maximum number of iterations for our neural network.
# If iterations are too low, the model underfits the data and becomes a very 
# poor model.
# On the other hand, excessively high number of iterations may not just cause
# overfitting, but also may drain excessive computing power and resources.


# ------------------------------------------------------------------------------
# Prediction and Evaluation
# ------------------------------------------------------------------------------

set.seed(10)
# Predict class probabilities
nn_probs <- predict(fit_neuralnetwork_model, x_test_neural_network_scaled, type = 
                      "raw")[, "Yes"]

# Convert probabilities to class labels (threshold = 0.5)
nn_class <- factor(
  ifelse(nn_probs > 0.5, "Yes", "No"),
  levels = c("No", "Yes")
)

# Confusion Matrix
confusionMatrix(nn_class, y_test_neural_network, positive = "Yes")

# The sensitivity is 54.29%, which means that the model can identify only
# 54.29% of true positives. This is the true positive rate, and is not too 
# satisfactory, but it is to be expected for imbalanced classes like the ones
# here.

# The specificity is 88.47%, which is quite good. This is the true negative rate
# and this means that the model can identify upto 88% of all true negatives.

# The accuracy is 79.4%, and the model is definitely better than our
# naive classification model.

# The 95% confidence interval for the accuracy score is (0.7761, 0.8111)
# This is a rather small aka tight interval, which is satisfactory.

# We notice that the performance of the neural network model is on par with the
# performance of our logistic regression model with p=0.5, however, the logistic
# regression model with the threshold maximising Youden's J statistic still
# seems to reign supreme when it comes to balancing sensitivity and specificity




# ROC and AUC
roc_nn <- roc(y_test_neural_network, nn_probs)
plot(roc_nn,main="ROC Curve",col="lightgoldenrod2",print.auc=TRUE)
auc(roc_nn) # 0.8301

# ------------------------------------------------------------------------------
# Model performance summary table
# All values taken from evaluated confusion matrices / ROC as above
# ------------------------------------------------------------------------------

# Let us create a simple table to visualise the key metrics of the models
# we have fit.

model_results <- data.frame(
  Model = c("Logistic (p = 0.5)","Logistic (p = 0.8)","Logistic (p = 0.4)",
"Logistic (Youden's J)","Ridge Regression","Lasso Regression","KNN (k = 25)",
"Neural Network"),
Sensitivity = c(0.5411,0.003571,0.6400,0.8232,0.4893,0.5089,0.5554,0.5429),
Specificity = c(0.9008,0.9981,0.8400,0.7126,0.9143,0.9104,0.8750,0.8847),
AUC = c(0.8453,0.5010,0.7411,0.7679,0.8469,0.8469,0.7152,0.8301))

print(model_results)

# ==============================================================================
# SECTION 6: CONCLUSION
# ==============================================================================
# This is the end of the coding section of this project

# Most of the methods, including ridge regression, lasso, KNN, Neural Networks,
# all seem to give results that are not too different from one another, 
# possibly indicating that it is how good models can get owing to the complexity
# of the dataset and the myriad of covariates involved.

# The logistic baseline model seems to have excellent predictive capacities.
# Our Neural Network model and the penalised models seem to have excellent
# predictive powers too.

# KNNs underperformed, which is indeed quite surprising.

# A brief conclusion we draw here is that the logistic regression model fitted
# according to maximisation of Youden's J statistic seems to be the best 
# performing model, which even outperforms deep learning method and other 
# machine learning methods when it comes to the sensitivity-specificity balance.

# Our analyses here are finance orientated, hence, we do not just pick the model
# with the largest AUC but we look at the models holistically.

# For understanding the analyses, explanations, conclusions drawn and decisions
# taken, please refer to the report attached with the submission.
# A more comprehensive conclusion is attached with the report too.


# Thank you very much.

# ==============================================================================
#                       END OF PROJECT

