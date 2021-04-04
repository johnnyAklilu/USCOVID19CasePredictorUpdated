
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('national-history-update.csv')

#print(df)

X = df[['day', 'totalTestResultsIncrease']].values
y = df['positiveIncrease'].values

print("US_COVID19Predictor")
model = LinearRegression()
model.fit(X, y)
ans = input("Do you want to see the pattern of the pandemic? ")
if ans=='y':
    plt.figure(figsize=(10, 6))
    plt.scatter(df['totalTestResultsIncrease'], df['positiveIncrease'])
    plt.title('Pattern of the Pandemic')
    plt.xlabel('totalTestResultsIncrease')
    plt.ylabel('positiveIncrease')
    plt.xlim(1, 2.0e+6)
    plt.ylim(0, 2.0e+6)
    plt.plot(X, model.predict(X), color='red')
    
    plt.show()
ans = input("Do you want to see the general developement of the pandemic day to day? ")
if ans == 'y':
    plt.scatter(df['day'], df['positiveIncrease'])
    plt.title('PositiveIncrease Per Day')
    plt.xlabel('day')
    plt.ylabel('positiveIncrease')
    plt.xlim(1, 5.0e+2)
    plt.ylim(0, 2.0e+6)
    plt.show()

#Slope Coefficient/ theta_1
print("Slope = " + str(model.coef_))


#Intercept
print("Intercept = " + str(model.intercept_))


#Prediction
print("Day 400, Among 1,232,995 test results "+str(model.predict([[400, 1232995]]))+" positive people.") #58702
print("Day 410, Among 1,406,795 test results "+str(model.predict([[410, 1406795]]))+" positive people.") #66836
print("Day 500, Among 1,609,100 test results "+str(model.predict([[500, 1609100]]))+" positive people.")
predicted_score = model.predict(X)
plt.plot(X, predicted_score)
plt.show()
#Score

print("Accuracy of the program = "+str(model.score(X, y)))


