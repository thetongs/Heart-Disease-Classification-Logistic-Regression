## Problem statement
# Patients has this disease or not prediction using logistic regression

## Libraries
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load dataset
#
dataset = pd.read_csv('heart.csv')
dataset.head()

## General informaion 
#
dataset.info()

## Statistical information
#
dataset.describe()

## Check for missing values
#
dataset.isna().sum()

## Data visualization for analysis
#
# bar plot age count
# using matplotlib
label = ['Male', 'Female']
data = list(dataset.sex.value_counts())

plt.barh(label, data, color = ['lightblue', 'lightpink'])
plt.title('Gender count comparision')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.show()
plt.close()


# bar plot age count
# using seaborn
#
import seaborn as sns

sns.barplot(label, data)
plt.title('Gender count comparision')
plt.xlabel('Count')
plt.ylabel('Gender')
plt.show()
plt.close()

# how age is spread and its reliability
#
data = dataset.age
data_mean = data.mean()
data_std = data.std()

bars = [data_std]
bar_category = ['age']
error_bars = [data_std]

plt.errorbar(bar_category, bars, yerr = error_bars,
             fmt = 'o', elinewidth = 3,
             alpha = 1, capsize = 5)
plt.title('error bar plot for age')
plt.xlabel('label')
plt.ylabel('std')
plt.show()
plt.close()

## Conclusion
# the SD bar is big so data are more variable from mean and less reliable

# target count
#
label = ['Have disease', 'not']
data = list(dataset.target.value_counts())

plt.barh(label, data, color = ['lightblue', 'lightpink'])
plt.title('Target count comparision')
plt.xlabel('Count')
plt.ylabel('Category')
plt.show()
plt.close()


## Percentage information
#
import math

male_cnt = dataset.sex.value_counts()[1]
total_cnt = dataset.sex.count()
female_per = 100 - (male_cnt/total_cnt) * 100
male_per = 100 - female_per
print("Percentage of Male : {} %".format(math.floor(male_per)))
print("Percentage of Female : {} %".format(math.ceil(female_per)))

have = math.floor((dataset.target.value_counts()[1] / dataset.target.count()) * 100)
not_have = 100 - have
print("Percentage of Have : {} %".format(math.floor(have)))
print("Percentage of Not have : {} %".format(not_have))


# box plot for age
data = dataset.age
box_plot_data = [data]

box = plt.boxplot(box_plot_data, patch_artist = True,
            labels = ['age'], notch = True, vert = 0, )

colors = ['lightgreen']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
plt.show()

## Conclusion
# It is good way to visualize data through their quartile.
# Least value is below 30.
# Lower quartile is above 45 and belwo 50.
# Median is near 55.
# Upper quartile is above 60 and below 65.
# Highest value is above 75.


# histogram for age
#
import math

def find_bins(X):
    N = len(X)
    K = 1 + 3.322 * math.log(55, 10)
    K = math.floor(K)
    return K

X = dataset.age
N = find_bins(X)

print("Bins required : {}".format(N))

plt.hist(X, bins = N, facecolor = 'lightgreen', width = 8)
plt.show()
plt.close()


# chest pain type - cp
#
category_names = ['0 type', '1 type', '2 type', '3 type']
category = dataset.cp.value_counts().sort_index().to_list()
plt.bar(category_names, category, color = 'rgbymck')
plt.xticks(color = 'green', rotation = 45, horizontalalignment = 'right')
plt.title('chest pain type and count')
plt.xlabel('category')
plt.ylabel('count')
plt.show()
plt.close()


# Disease with age and target
#
pd.crosstab(dataset.age,dataset.target).plot(kind="bar",figsize=(22,6))
plt.title('Disease and Age and target')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

## Dependent and Independent variables
#

X = dataset.iloc[:,:-1]
Y = dataset.iloc[:,-1]


## Splitting 
#
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#
X_train.head()

#
X_test.head

#
Y_train.head()

# 
Y_test.head()

## Scaler need to scale data into one format
#
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

## Lets try logistic regression classification
#
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,Y_train)

## Prediction
#
Y_pred = lr.predict(X_test)

## Check corrections
#
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(Y_test,Y_pred)
cm_lr


## Find accuracy
#
accuracy = lr.score(X_test, Y_test)
accuracy

## Predictions
#
lr.predict(sc_x.transform([[30, 0, 2, 122, 213, 0, 1, 165, 0, 0.2, 1, 0, 2]]))




