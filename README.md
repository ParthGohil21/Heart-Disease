# Heart-Disease
![image](https://user-images.githubusercontent.com/85442734/161431719-7c412367-c441-424e-a94b-6a15bb7b0656.png)


## Dataset Overview
Originally, the dataset come from the CDC and is a major part of the Behavioral Risk Factor 
Surveillance System (BRFSS), which conducts annual telephone surveys to gather data on the health status of U.S. residents.


## Data Description 
       
1. HeartDisease :Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) 
2. BMI: Body Mass Index (BMI)
3. Smoking: Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]
4. AlcoholDrinking: Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
5. Stroke: (Ever told) (you had) a stroke?
6. PhysicalHealth: Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30
7. MentalHealth: Thinking about your mental health, for how many days during the past 30 days was your mental health not good?
8. DiffWalking: Do you have serious difficulty walking or climbing stairs?
9. Sex: Are you male or female?
10. AgeCategory: Fourteen-level age category
11. Race: Imputed race/ethnicity value
12. Diabetic: (Ever told) (you had) diabetes?
13. PhysicalActivity: Adults who reported doing physical activity or exercise during the past 30 days other than their regular job
14. GenHealth: Would you say that in general your health is..
15. SleepTime: On average, how many hours of sleep do you get in a 24-hour period?
16. Asthma:(Ever told) (you had) asthma?
17. KidneyDisease: Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?
18. SkinCancer : (Ever told) (you had) skin cancer?


## Project Overview
The dataset contains 18 variables (9 booleans, 5 strings and 4 decimals). In machine learning projects, "HeartDisease" can be used as the explonatory variable, but note that the classes are heavily unbalanced. So I used Over Sampling for balancing classes. After balancing dataset I have used Light GBM Classifier to predict wheather the person is likely to have heart disease or not('Yes' if person is likely to have heart disease and 'No' If person isn't likely to have heart disease).


## Tools Overview 
The following are the tools that are covered in the notebooks. They are popular tools that machine learning engineers and data scientists need in one way or another and day to day.

Python is a high level programming language that has got a lot of popularity in the data community and with the rapid growth of the libraries and frameworks, this is a right programming language to do ML.

NumPy is a scientific computing tool used for array or matrix operations.

Pandas is a great and simple tool for analyzing and manipulating data from a variety of different sources.

Matplotlib is a comprehensive data visualization tool used to create static, animated, and interactive visualizations in Python.

Seaborn is another data visualization tool built on top of Matplotlib which is pretty simple to use.

Scikit-Learn: Instead of building machine learning models from scratch, Scikit-Learn makes it easy to use classical models in a few lines of code. This tool is adapted by almost the whole of the ML community and industries, from the startups to the big techs.

Lightgbm is widely used for big dataset. It is faster than most of the tree based algorithms and also provides easy to optimize structure. 
