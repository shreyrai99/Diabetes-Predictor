# Diabetes Predictor

  This Project was made to make accurate predictions about whether a person has Diabetes or not by using Machine Learning Algorithms.
  I deployed the Machine Learning model using Flask API. I trained various machine learning algorithms and finally used Gradient Boost
  for the project as it gave the best results.
  
  
  The Model is deployed on Heroku at the following link:
  
  https://diabetes-prediction-by-shrey.herokuapp.com/
  
 
# Technologies Used: #
  I used the following tools and technologies in order to successfully make the project: <br />
    1. Python <br />
    2. HTML <br />
    3. CSS <br />
    4. File Handling technique <br />
    5. Flask (Framework) <br />
    6. sklearn library for implementing various Machine learning algorithms <br />
    
   
  # Project Structure #
   This project has four major parts :
   1. model.py : This contains code fot our Machine Learning model to predict the chances of a patient having Diabetes. 
   2. app.py: This contains Flask APIs that receives patient's details through GUI or API calls, computes the precited value based on      our model and predicts whether the patient has Diabetes or not.
   3. templates: This folder contains the HTML template to allow user to enter various fields inorder to know his chances of having        disease. 
   4. static: It contains the CSS part of the project. 
   
    
  # Running the Model: #
   1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
   python model.py 
   
   
   This would create a serialized version of our model into a file model.pkl 
   2. Run app.py using below command to start Flask API 
   python app.py <br />
   By default, flask will run on port 5000. 
   3. Navigate to URL http://localhost:5000   
   4. We will see the below home page where we have to enter the details. 
   
   
   ![Home_Page](https://user-images.githubusercontent.com/51885421/89995555-3ed4bd00-dca7-11ea-9e69-142ec9502327.png)
   
   
   5. The out put will be 0 or 1. <br />
      1 means the person is having Diabetes <br />
      0 means he is safe  <br />
   
   
   6. Demonstartion For Patients having Diabetes 
   
   ![patient1_data](https://user-images.githubusercontent.com/51885421/89995572-44ca9e00-dca7-11ea-85c3-34db6481f355.png)
   
   
   ![patient1_result](https://user-images.githubusercontent.com/51885421/89997140-5dd44e80-dca9-11ea-92bd-4f107d6a0d6d.png)
   
   
   
   7. Demonstartions For Patients Not having Diabetes
   
   ![patient2_data](https://user-images.githubusercontent.com/51885421/89995602-4f853300-dca7-11ea-9faa-f22a8483520f.png)
   
   
   ![patient2_result](https://user-images.githubusercontent.com/51885421/89997156-63ca2f80-dca9-11ea-9620-16b91f4fc1d0.png)
   
   
   8. Do visit my Project on predicting chances of having Heart Disease by visiting the following link:
   
   
   https://github.com/shreyrai99/Heart-Disease-Predictor
   
   
   
   # Thank You ! #
 
