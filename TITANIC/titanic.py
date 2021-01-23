# =========================== IMPORT ==========================================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# =========================== VISUALISATION ===================================
data = pd.read_csv("train.csv")
data.head()
data["Age"].mode()
data["Age"].mean()
data["Age"].median()


# =============================================================================
# =========================== BUILD DATA train/test ===========================
# =============================================================================

# =============== data_train
data_train = pd.read_csv("train.csv")
column_input = ["Sex","Age","Embarked","Pclass"]
column_class = ["Survived"]
column_keep = column_input+column_class
data_train = data_train.dropna(subset=column_keep)

data_train_Input = data_train[column_input]
data_train_Input = pd.get_dummies(data_train_Input)
data_train_Class = data_train[column_class]

# =============== data_test
data_test = pd.read_csv("test.csv")
data_test_Input = pd.get_dummies(data_test[column_input])
data_test_Input = data_test_Input.fillna(data_test_Input.mean())

# =============================================================================
# ================================= BUILD MODEL ===============================
# =============================================================================
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(data_train_Input, data_train_Class)
predictions = model.predict(data_test_Input)

output = pd.DataFrame({'PassengerId': data_test.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")