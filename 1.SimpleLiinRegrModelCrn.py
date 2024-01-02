# library used for working with data sets. It has functions for analyzing, cleaning, exploring, and manipulating data
import pandas as p 
# retrieve and store data from the sheet
dataset = p.read_csv("Salary_Data.csv")
#print(dataset)
yoe = dataset[["YearsExperience"]]   #independant - Input value
sal = dataset[["Salary"]]  #dependant - Output expected
#print(yoe)
#print(sal)
# sklearn - module provides Simple & efficient tools for predictive data analysis for ML
from sklearn.model_selection import train_test_split  #procedure
# splitting dataset in to 4 parameters for train and test the model.
# test size - % of data used to train and test from the whole
x_train,x_test,y_train,y_test = train_test_split(yoe,sal,test_size=0.30,random_state=0)

# Linear regression - data-points to draw a straight line used to predict future values
from sklearn.linear_model import LinearRegression   #procedure
reg_or = LinearRegression() #only 11 and 12 code will change for diff algorithms
reg_or.fit(x_train.values,y_train.values)  #weight and bias is calculated

weight = reg_or.coef_
bias = reg_or.intercept_
#print(weight)
#print(bias)
# Passing inputs to process prediction
predict = reg_or.predict(x_test.values)  #y_test is actual data to compare the result

from sklearn.metrics import r2_score
# R2 is validating the prediction score between 0 and 1.
r_score = r2_score(y_test,predict) 
print(r_score) # good model score should be nearest to 1.

#Saving the model
import pickle # converting an object into character/byte stream
file = "SalPredModel.sav" # file name
# save the model into a file
pickle.dump(reg_or,open(file,"wb")) # wb - write binary
print("Model saved")
#---------------------------------------------------------

#Loading the saved model to read and check or deployment
load_Model = pickle.load(open("SalPredModel.sav","rb")) #rb - read library
print("Model loaded")
result = load_Model.predict([ [15] ])
print("load check result:",result)