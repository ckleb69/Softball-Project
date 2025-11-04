import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


df = pd.read_csv(r"C:\Users\Connor\Downloads\CleanSZ.csv")
df = df[(df['PitchCall'] == 'InPlay') & df['ExitSpeed'].notna() & df['Angle'].notna()]

X = df[['ExitSpeed', 'Angle']]
y = df['Outcome']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Balance classes
sm = SMOTE(sampling_strategy={
    'Out': 1500, 'Single': 1500, 'XBH': 1500, 'Homerun': 1500
}, random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

#Train
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.1, random_state=42, stratify=y_res
)

# k and weighting
param_grid = {
    'n_neighbors': [6],
    'weights': ['uniform', 'distance']
}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='f1_weighted')
grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)

#Evaluate
knn = grid.best_estimator_
y_pred = knn.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

#probabilities
probs = knn.predict_proba(X_test)
prob_df = pd.DataFrame(probs, columns=knn.classes_)
print("probabilities of", prob_df.head())
print("best neighbors (k):", grid.best_params_['n_neighbors'])

#Cross-validation
cv_scores = cross_val_score(knn, X_res, y_res, cv=10, scoring='f1_weighted')
print("Mean CV F1:", cv_scores.mean(), "±", cv_scores.std())
