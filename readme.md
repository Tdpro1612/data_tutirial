## step 1 : get data

```
from google.colab import drive
drive.mount("/content/drive")
```
Mounted at /content/drive
```
file_name = "/content/kaggle.json"
with open(file_name, 'r') as f:
    document =  js.loads(f.read())
    
print(document)
#{'key': 'c00046e4b093c67705ca6d402f00aa31', 'username': 'nguynnguynkhoa'}

os.environ['KAGGLE_USERNAME'] = document['username']
os.environ['KAGGLE_KEY'] = document['key']

#get API
```
{'username': 'nguynnguynkhoa', 'key': '86d54991913d6b87880d297405c8173e'}
```
!kaggle datasets download -d uciml/adult-census-income
```
Downloading adult-census-income.zip to /content
  0% 0.00/450k [00:00<?, ?B/s]
100% 450k/450k [00:00<00:00, 67.7MB/s]
```
!unzip /content/adult-census-income.zip
```
Archive:  /content/adult-census-income.zip
  inflating: adult.csv               
```
df = pd.read_csv("/content/adult.csv")
```

## step 2 : Problem description
![anh1](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh1.jpg)

you see this data not null file

![anh2](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh2.jpg)
![anh3](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh3.jpg)
![anh4](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh4.jpg)
![anh5](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh5.jpg)
![anh6](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh6.jpg)
![anh7](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh7.jpg)
![anh8](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh8.jpg)
![anh9](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh9.jpg)
![anh10](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh10.jpg)
![anh11](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh11.jpg)

But this data have some missing values "?"

## step3 : the exploratory data analyst

![anh14](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh14.jpg)

the data have age of values between from 20-60,the values of 60+ is very few

![anh15](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh15.jpg)

the num of education is 9,10,13 take up large numbers

![anh16](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh16.jpg)

the white ,black skin is very much

![anh18](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh18.jpg)


the most of the hours per week is 40


![anh19](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh19.jpg)

the values of income have more than 50k focus on 37 to 59

## step 4 : preprocessing data
![anh12](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh12.jpg)

you see this data not null file but this have some value is "?" 
So we fill it by mod value because the values of data is imbalance and it have ratio is very small
```
df = df.replace('?', np.nan)
nan_cols = [i for i in df.columns if df[i].isnull().any()]
for col in nan_cols:
  df[col].fillna(df[col].mode()[0], inplace=True)
```
#### label all feature
```
from sklearn.preprocessing import LabelEncoder
for col in df.columns:
  if df[col].dtypes == 'object':
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
```
#### chose some feature have affection of income
```
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
plt.style.use('default')
plt.figure(figsize=(17, 8))
ax = sns.heatmap(corr,cmap='RdYlGn', mask=mask, vmax=.3,annot=True)
plt.show()
```

![anh21](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh22.jpg)

you can see have 8 feature have affection of income : 'age','education','education.num','race','sex','capital.gain','capital.loss','hours.per.week' (>0.05)

Scale all features

```
from sklearn.preprocessing import StandardScaler
for col in X.columns:
  scaler = StandardScaler()
  X[col] = scaler.fit_transform(X[col].values.reshape(-1, 1))
```

```
round(Y.value_counts(normalize=True) * 100, 2).astype(str) + "%"
```

![anh13](https://github.com/Tdpro1612/tutorial_data_science/blob/f384eb9f4916096c0123f19661acad7151a5a67e/dt/anh13.jpg)

the rate of label is imbalance,we use random Over sample to get more the values of data is less than

```
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=101)
ros.fit(X,Y.values.ravel())

X_resampled, Y_resampled = ros.fit_resample(X,Y.values.ravel())

X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

Y_resampled = pd.DataFrame(Y_resampled, columns=Y.columns)
```
```
round(Y_resampled.value_counts(normalize=True) * 100, 2).astype(str) + "%"
```
![anh23](https://github.com/Tdpro1612/tutorial_data_science/blob/a4f2fcd0f64b7c34263b95f4bb4cb40e4469f9f2/dt/anh23.jpg)

the rate of label is balance

**Now,We can train !!!**
## step 5 : train model

we build all model to see this
- logistic regression
```
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state=101)
```
```
log_reg.fit(X_train,y_train.values.ravel())
```
```
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=101, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
```
```
y_pred_log_reg = log_reg.predict(X_test)
```

- KNN classifier
```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train.values.ravel())
```
```
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
```
```
y_pred_knn = knn.predict(X_test)
```
- Support vector Classifier
```
from sklearn.svm import SVC
svc = SVC(random_state=101)
svc.fit(X_train,y_train.values.ravel())
```
```
SVC(C=1.0, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=101, shrinking=True, tol=0.001,
    verbose=False)
```
```
y_pred_svc = svc.predict(X_test)
```
- Navie Bayes Classifier
```
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(X_train,y_train.values.ravel())
```
```
GaussianNB(priors=None, var_smoothing=1e-09)
```
```
y_pred_nb = nb.predict(X_test)
```
- Decision Tree Classifier
```
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(random_state=101)
dec_tree.fit(X_train,y_train)
```
```
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=101, splitter='best')
```
```
y_pred_dec_tree = dec_tree.predict(X_test)
```
- Random Forest Classifier
```
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=101)
rfc.fit(X_train,y_train.values.ravel())
```
```

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=101)
rfc.fit(X_train,y_train.values.ravel())
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=101,
                       verbose=0, warm_start=False)
```
```
y_pred_rfc = rfc.predict(X_test)
```

## step 6 : Accuracy,f1 score
```
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
```
- LogisticRegression
```
print("LogisticRegression")
print("accuracy",round(accuracy_score(y_test,y_pred_log_reg)*100,2))
print("f1",round(f1_score(y_test,y_pred_log_reg)*100,2))
```
```
LogisticRegression
accuracy 76.16
f1 75.81
```
- KNeighborsClassifier
```
print("KNeighborsClassifier")
print("accuracy",round(accuracy_score(y_test,y_pred_knn)*100,2))
print("f1",round(f1_score(y_test,y_pred_knn)*100,2))
```
```
KNeighborsClassifier
accuracy 79.19
f1 79.9
```
- Support vector Classifier
```
print("Support vector Classifier")
print("accuracy",round(accuracy_score(y_test,y_pred_svc)*100,2))
print("f1",round(f1_score(y_test,y_pred_svc)*100,2))
```
```
Support vector Classifier
accuracy 77.62
f1 78.36
```
- Navie Bayes Classifier
```
print("Navie Bayes Classifier")
print("accuracy",round(accuracy_score(y_test,y_pred_nb)*100,2))
print("f1",round(f1_score(y_test,y_pred_nb)*100,2))
```
```
Navie Bayes Classifier
accuracy 63.67
f1 47.27
```
- Decision Tree Classifier
```
print("Decision Tree Classifier")
print("accuracy",round(accuracy_score(y_test,y_pred_dec_tree)*100,2))
print("f1",round(f1_score(y_test,y_pred_dec_tree)*100,2))
```
```
Decision Tree Classifier
accuracy 82.66
f1 83.14
```
- Random Forest Classifier
```
print("Random Forest Classifier")
print("accuracy",round(accuracy_score(y_test,y_pred_rfc)*100,2))
print("f1",round(f1_score(y_test,y_pred_rfc)*100,2))
```
```
Random Forest Classifier
accuracy 83.32
f1 83.92
```

## step6 : Hyperparameter tuning
```
from sklearn.model_selection import RandomizedSearchCV
```
```
n_estimators = [int(x) for x in np.linspace(start=40,stop=150,num=15)]
max_depth = [int(x) for x in np.linspace(start=40,stop=150,num=15)]
```
```
para_dict = {
    'n_estimators' : n_estimators,
    'max_depth' : max_depth
}
```
```
rf_tune = RandomForestClassifier(random_state=101)
rf_cv = RandomizedSearchCV(estimator=rf_tune, param_distributions=para_dict, cv=5, random_state=101)
rf_cv.fit(X_train,y_train.values.ravel())
```
```
RandomizedSearchCV(cv=5, error_score=nan,
                   estimator=RandomForestClassifier(bootstrap=True,
                                                    ccp_alpha=0.0,
                                                    class_weight=None,
                                                    criterion='gini',
                                                    max_depth=None,
                                                    max_features='auto',
                                                    max_leaf_nodes=None,
                                                    max_samples=None,
                                                    min_impurity_decrease=0.0,
                                                    min_impurity_split=None,
                                                    min_samples_leaf=1,
                                                    min_samples_split=2,
                                                    min_weight_fraction_leaf=0.0,
                                                    n_estimators=100,
                                                    n_jobs...
                                                    oob_score=False,
                                                    random_state=101, verbose=0,
                                                    warm_start=False),
                   iid='deprecated', n_iter=10, n_jobs=None,
                   param_distributions={'max_depth': [40, 47, 55, 63, 71, 79,
                                                      87, 95, 102, 110, 118,
                                                      126, 134, 142, 150],
                                        'n_estimators': [40, 47, 55, 63, 71, 79,
                                                         87, 95, 102, 110, 118,
                                                         126, 134, 142, 150]},
                   pre_dispatch='2*n_jobs', random_state=101, refit=True,
                   return_train_score=False, scoring=None, verbose=0)
```
```
rf_cv.best_score_
# 0.8302739212679107
```
```
rf_cv.best_params_
# {'max_depth': 63, 'n_estimators': 110}
```
```
rf_best = RandomForestClassifier(max_depth=63,n_estimators=110,random_state=101)
rf_best.fit(X_train,y_train.values.ravel())
```
```
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=63, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=110,
                       n_jobs=None, oob_score=False, random_state=101,
                       verbose=0, warm_start=False)
```
```
y_pred_best = rf_best.predict(X_test)
```
```
print("Random Forest Classifier")
print("accuracy",round(accuracy_score(y_test,y_pred_best)*100,2))
print("f1",round(f1_score(y_test,y_pred_best)*100,2))
```
```
Random Forest Classifier
accuracy 83.32
f1 83.92
```
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred_best)
```
```
plt.style.use('default')
sns.heatmap(cm, annot=True, fmt='d',cmap='YlGnBu')
plt.savefig('heatmap.png')
plt.show()
```
![heatmap](https://github.com/Tdpro1612/tutorial_data_science/blob/b69ad9faf6405bebcc9047a6ebd2677c90d5a97d/dt/heatmap.png)

**Wow,We have completed a tutorial of data science**
