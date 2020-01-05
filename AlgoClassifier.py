import pandas as pd

train = pd.read_csv("dataset/dataset/train.csv")

test = pd.read_csv("dataset/dataset/test.csv")

df_id = test['ID']
train.drop('ID', axis=1, inplace=True)
test.drop('ID', axis=1, inplace=True)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train.drop('Result', axis=1))
scaled = scaler.transform(train.drop('Result', axis=1))

df = pd.DataFrame(scaled, columns=train.columns[:-1])

X = df
Y = train['Result']
pca = PCA(n_components=25)
X_final = pca.fit_transform(X)

#from sklearn.model_selection import train_test_split
#train_x, test_x, train_y, test_y = train_test_split(X_final, Y, test_size=0.3, random_state=42)

train_x = X_final
train_y = Y

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(train_x, train_y)


scaler1 = StandardScaler()
scaler1.fit(test)
scaled1 = scaler1.transform(test)

df_test = pd.DataFrame(scaled1, columns=test.columns)
pca = PCA(n_components=25)
Test_final = pca.fit_transform(df_test)


predictions = rfc.predict(Test_final)
df_pred = pd.DataFrame(predictions, columns=['Result'])
print(df_pred.head())
#from sklearn.metrics import classification_report

#print(classification_report(test_y, df_pred))

result = pd.concat([df_id, df_pred], axis=1)
print(result.head())

result.to_csv('submission.csv', index=False)
