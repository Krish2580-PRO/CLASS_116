import pandas as pa
import plotly.express as pe
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------------------------------

data = pa.read_csv("project.csv")

toefl_score = data["TOEFL Score"].tolist()
gre_score = data["GRE Score"].tolist()

result = data["Chance of admit"].tolist()
fig = pe.scatter(x = toefl_score, y = result)
# fig.show()

colors = []

for i in result:
    if i == 1:
        colors.append("green")
    else:
        colors.append("red")

fig = go.Figure(data= go.Scatter(x = toefl_score, y = gre_score ,mode="markers" ,  marker = dict(color = colors)))
fig.show()

# ------------------------------------------------------------

#Taking together Age and Salary of the person
factors = data[["TOEFL Score", "GRE Score"]]

#Purchases made
result_ = data["Chance of admit"]

toefl_train, toefl_test, result_train, result_test = train_test_split(factors, result_, test_size = 0.25, random_state = 0)
print(toefl_train[0:5])

# -----------------------------------------------------

sc_x = StandardScaler() 

toefl_train = sc_x.fit_transform(toefl_train)  
toefl_test = sc_x.transform(toefl_test) 

print("--------------------------------")
print(toefl_train[0:5])


# ----------------------------------------------------------

# random_state = 0 --> passing Training data 
# random_state = 1 --> passing Testing data 

lr = LogisticRegression(random_state = 0) 
lr.fit(toefl_train, result_train)

result_pred = lr.predict(toefl_test)

print("--------------------------------")
print ("Accuracy : ", accuracy_score(result_test, result_pred)) 


# ---------------------------------------------------------------------------

print("------------------------------------------------")
user_toefl_score = int(input("Enter the toefl score of the user :- "))
user_gre_score = int(input("Enter the gre score of the user :- "))

user_test = sc_x.transform([[user_gre_score , user_toefl_score]])

user_pred = lr.predict(user_test)

if user_pred[0] == 1:
    print("Might get admitted !! ")
else:
    print("Might NOT get admitted!! ")


















