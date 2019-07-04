# Food-Recommendation-System

Abstract
Inadequate and inappropriate intake of food is known to cause various health issues and diseases. Due to lack of concise information about healthy diet, people have to rely on medicines instead of taking preventive measures in food intake. Due to diversity in food components and large number of dietary sources, it is challenging to perform real-time selection of diet patterns that must fulfill one’s nutrition needs.
In this project,a total number of 101 types of cuisines are taken into account.According to some vital human parameters for example weight,height,body mass index,gender,user is suggested which cuisine he should take.

Documentation
1)Calorie_Calculation

bmi_level.csv: Data consist of obesity level on the basis of height weight and gender
calorie_data.csv: Data consist of calories of 101 food categories with weights in grams
recommendation_gui.py: File which take image to prediction.Implemented some part of GUI.
predict_food_live.py: Live food detecion
haarcascade_frontalface_default.xml: open cv front face classifier
foodset:-It contains some images of food,model used here will predict a perticular cuisine from these images.
food_detect_model :- Download this model from the given link.
https://drive.google.com/file/d/13hXkicfY_9y0zEdYVEL3OYQ8XHLJ-RVR/view?usp=sharing

2)Food_Recommendation
details_7.csv: It consist of customer information
fod_recmond_dbase.py: It ecommends food to the customer.
store_order_signup.py: Store details of food in database for old customer and also details of new customer
haarcascade_frontalface_alt2.xml:Opencv classifier to detect image
food_detect_model :- Download this model from the given link.
https://drive.google.com/file/d/13hXkicfY_9y0zEdYVEL3OYQ8XHLJ-RVR/view?usp=sharing

Dataset Link:-https://www.vision.ee.ethz.ch/datasets_extra/food-101/


Credits must be given to the company who created Food 101 dataset and my mentor Mr. Saurabh Bhardwaj(https://www.linkedin.com/in/saurabh-bhardwaj-b97a1539/).
