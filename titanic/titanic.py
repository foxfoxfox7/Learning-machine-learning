import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import tree
from scipy import stats


def simple_bar_comparison(df, x, hue):
    sns.countplot(data = df, x = x, hue = hue)
    plt.show()

def print_answer(name, answer):
    if answer == 0:
        print(name + ' - Died')
    else:
        print(name + ' - Survived')

def make_test_split(df, drop, ts = 0.3):
    copy_df = df.copy()
    y = copy_df[drop].values
    copy_df = copy_df.drop(drop, axis = 1)
    X = copy_df.values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size =  ts, random_state = 0
        )

    return X_train, X_test, y_train, y_test


#gender_df = pd.read_csv('gender_submission.csv')
#print(gender_df.head())
#print(gender_df.shape)

test_df = pd.read_csv('test.csv')
print(test_df.head())
print(test_df.info())
test_start = test_df.copy()

train_df = pd.read_csv('train.csv')
print(train_df.head())
print(train_df.info())

# Printing out the missing data information
null_bool = pd.isnull(train_df)
print(null_bool.sum())

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].median())
#train_df['Age'] = train_df['Age'].interpolate()
#test_df['Age'] = test_df['Age'].interpolate()
#test_df['Fare'] = test_df['Fare'].interpolate()

# Making new dataframes without the rows that have missing values
cabin_df = train_df[null_bool['Cabin'] == False]
# Adding a column with information from an existing column
cabin_df['Cabin_letter'] = cabin_df['Cabin'].str[0]
sns.countplot(data=cabin_df, x='Cabin_letter', hue='Survived')
#plt.show()

train_df = train_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
test_df = test_df.drop(['Name', 'Ticket', 'Cabin'], axis = 1)
train_df = train_df.set_index('PassengerId')
test_df = test_df.set_index('PassengerId')
print(train_df.head())

########################################################################
# basic visualization
########################################################################
'''
bar_plot_list = ['Sex', 'Pclass', 'SibSp', 'Parch', 'Embarked']
for col in bar_plot_list:
    simple_bar_comparison(df = train_df, x = col, hue = 'Survived')

# Basic histogram plots
fig, ax = plt.subplots()
sns.distplot(train_df['Age'].where(train_df['Survived'] == 1),
    label = 'survived')
sns.distplot(train_df['Age'].where(train_df['Survived'] == 0),
    label = 'not survived')
ax.set(xlim = (0, None))
plt.show()

# Two subplots next to each other
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,)#sharey = True,  figsize=(7,4)
sns.distplot(train_df['Fare'].where(train_df['Survived'] == 1),
    label = 'survived', kde = False, ax = ax0)
sns.distplot(train_df['Fare'].where(train_df['Survived'] == 0),
    label = 'not survived', kde = False, ax = ax0)
sns.distplot(train_df['Fare'].where(train_df['Survived'] == 1),
    label = 'survived', kde = False, ax = ax1)
sns.distplot(train_df['Fare'].where(train_df['Survived'] == 0),
    label = 'not survived', kde = False, ax = ax1)
ax1.set(xlim = (0, 200), ylim = (0, 100))
ax0.set(xlim = (0, None))
ax1.legend()
plt.show()
'''
########################################################################
# ML setup
########################################################################

mac_learn_df = train_df.copy()

dummies = []
dummies_test = []
dummy_cols = ['Pclass', 'Sex', 'Embarked']
for col in dummy_cols:
    dummies.append(pd.get_dummies(mac_learn_df[col]))
    dummies_test.append(pd.get_dummies(test_df[col]))

titanic_dummies = pd.concat(dummies, axis = 1)
titanic_dummies_test = pd.concat(dummies_test, axis = 1)

new_col_names = ['Pclass1', 'Pclass2', 'Pclass3', 'female',
 'male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
titanic_dummies.columns = new_col_names
titanic_dummies_test.columns = new_col_names

mac_learn_df = mac_learn_df.drop(dummy_cols, axis = 1)
test_df = test_df.drop(dummy_cols, axis = 1)

ml_dummy_df = pd.concat((mac_learn_df, titanic_dummies), axis = 1)
test_final_df = pd.concat((test_df, titanic_dummies_test), axis = 1)
print(ml_dummy_df.head())
print(ml_dummy_df.info())

########################################################################
# Benchmark score
########################################################################

# Gets the values of the dropped category and the values of the rest and
# makes a test train split out of it
X_train, X_test, y_train, y_test = make_test_split(ml_dummy_df, 'Survived')
print(ml_dummy_df.columns)
print(ml_dummy_df.head())

#print('y - ', y.shape)
#print('x - ', x.shape)
#print('x_train - ', x_train.shape)
#print('x_test - ', x_test.shape)
#print('y_train - ', y_train.shape)
#print('y_test - ', y_test.shape)

clf = tree.DecisionTreeClassifier(max_depth=4)
clf.fit(X_train,y_train)
print('\nSCORE - ', clf.score(X_test,y_test), '\n')
#print('n leaves - ', clf.get_n_leaves())

#test_answers = clf.predict(test_final_df)
#for i in range(10):
#    print_answer(test_start['Name'][i], test_answers[i])

features = ml_dummy_df.columns.tolist()
del features[0]

for i, col in enumerate(features):
    print(col, ' - ', clf.feature_importances_[i])

########################################################################
# Round 2
########################################################################
'''
# correlation between a bivariate categorical val and a continuous val
print('C - ', stats.pointbiserialr(ml_dummy_df['Embarked_C'], ml_dummy_df['Fare']))
print('Q - ', stats.pointbiserialr(ml_dummy_df['Embarked_Q'], ml_dummy_df['Fare']))
print('S - ', stats.pointbiserialr(ml_dummy_df['Embarked_S'], ml_dummy_df['Fare']))



ml_dummy_df = ml_dummy_df.drop(['Embarked_C', 'Embarked_Q', 'Embarked_S'], axis = 1)

X_train2, X_test2, y_train2, y_test2 = make_test_split(ml_dummy_df, 'Survived')

clf2 = tree.DecisionTreeClassifier(max_depth=4)
clf2.fit(X_train2, y_train2)
print('\nSCORE - ', clf2.score(X_test2, y_test2), '\n')

'''
