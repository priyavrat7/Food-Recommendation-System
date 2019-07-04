#--------------------------GUI---------------------------
from tkinter import *
from tkinter import ttk
import pandas as pd
global food_btn
food_btn = ''
def func():
    global food_btn
    l2.configure(text=cb.get())
    food_btn = cb.get()
    #print(cb.get())

def close_window (): 
    win.destroy()
win =Tk()
win.geometry('400x300')

df=pd.read_csv('calorie_data.csv')
cuisines=list(df['categories'].values)
l1=Label(win,text="Choose Your Favourite Food")
l1.grid(column=0, row=1)
cb=ttk.Combobox(win,values=cuisines,width=16)
cb.grid(column=0, row=2)
cb.current(0)
b=Button(win,text="Click Here To select",command=func)
b.grid(column=1, row=2)
l2=Label(win,text="")
l2.grid(column=0, row=3)
button = Button (win, text = "Close", command = close_window)
button.grid(column=1, row=3)
#button.pack()
win.mainloop()
#--------------------------GUI---------------------------
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import tensorflow.keras.backend as K
#import shlex e=shelx.split(file_read)
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow
#To ignore tensorflow warning
tensorflow.logging.set_verbosity(tensorflow.logging.ERROR)

K.clear_session()
# Load the model
km = load_model(r'food_detect_model.hdf5',compile = False)
#Read the calorie and weight dataset 
df=pd.read_csv('calorie_data.csv')
# e is denoted to the list of Food 101 items
#e=['Apple pie', 'Baby back ribs', 'Baklava', 'Beef carpaccio', 'Beef tartare', 'Beet salad', 'Beignets', 'Bibimbap', 'Bread pudding', 'Breakfast burrito', 'Bruschetta', 'Caesar salad', 'Cannoli', 'Caprese salad', 'Carrot cake', 'Ceviche', 'Cheesecake', 'Cheese plate', 'Chicken curry', 'Chicken quesadilla', 'Chicken wings', 'Chocolate cake', 'Chocolate mousse', 'Churros', 'Clam chowder', 'Club sandwich', 'Crab cakes', 'Creme brulee', 'Croque madame', 'Cup cakes', 'Deviled eggs', 'Donuts', 'Dumplings', 'Edamame', 'Eggs benedict', 'Escargots', 'Falafel', 'Filet mignon', 'Fish and chips', 'Foie gras', 'French fries', 'French onion soup', 'French toast', 'Fried calamari', 'Fried rice', 'Frozen yogurt', 'Garlic bread', 'Gnocchi', 'Greek salad', 'Grilled cheese sandwich', 'Grilled salmon', 'Guacamole', 'Gyoza', 'Hamburger', 'Hot and sour soup', 'Hot dog', 'Huevos rancheros', 'Hummus', 'Ice cream', 'Lasagna', 'Lobster bisque', 'Lobster roll sandwich', 'Macaroni and cheese', 'Macarons', 'Miso soup', 'Mussels', 'Nachos', 'Omelette', 'Onion rings', 'Oysters', 'Pad thai', 'Paella', 'Pancakes', 'Panna cotta', 'Peking duck', 'Pho', 'Pizza', 'Pork chop', 'Poutine', 'Prime rib', 'Pulled pork sandwich', 'Ramen', 'Ravioli', 'Red velvet cake', 'Risotto', 'Samosa', 'Sashimi', 'Scallops', 'Seaweed salad', 'Shrimp and grits', 'Spaghetti bolognese', 'Spaghetti carbonara', 'Spring rolls', 'Steak', 'Strawberry shortcake', 'Sushi', 'Tacos', 'Takoyaki', 'Tiramisu', 'Tuna tartare', 'Waffles']

#When user will not select any of the food category the value of food_btn will not
#be modified,globally it id a null string
if (food_btn == ''):
    print("You didn't choose food a category.Try Again.......")
    exit(1)
else:
    print("Calculating calories of category:- "+food_btn)
e=list(df['categories'].values)


read_food = "foodset/"+food_btn+".jpg"
im_1=cv2.imread(read_food)
#remove face from image

#When image of the selected categoryis inavailable, variable im_1 would be None
if im_1 is None:
    print("No image available for this category.Try Again.......")
    exit(1)
gray = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")
fd = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
)
for (x, y, w, h) in fd:
    cv2.rectangle(im_1, (x, y), (x + w, y + h), (0, 0, 0), 2)
    im_1[y:y+h,x:x+w]=0
#Convert the image and perform operation so every value of array comes in 0 to 1 

roi=cv2.resize(im_1,(299, 299))
roi=img_to_array(roi)
roi=np.expand_dims(roi,axis=0)
roi /= 255.
#predict the food by the help of model
pred=km.predict(roi)
#print(pred)
max_e=max(pred)
ind=pred.argmax()
pred_value=e[ind]
#print(e[ind])

#Calculating calories and weight of food
print('Calories of food are:-',end=' ')
print(df[df['categories']==pred_value ]['calories'].values[0])

print("Its weight in grams:-",end=' ')
print(df[df['categories']== pred_value]['weight'].values[0])
#Now we sugest the food to person by calculating obesity level form height,weight and gender
calo_per_wght=df[df['categories']== pred_value]['cal_per_weight'].values[0]
height=int(input("Enter your height:-"))
weight= int(input("Enter your weight:-")) 
gender= input("Enter your gender M for Male and F for Female:-")
if gender=='M':
           gender_no=0
else:
           gender_no=1
#making datframe
d = {'Height': [height], 'Weight': [weight],'Gender_no':[gender_no]}
df1 = pd.DataFrame(data=d)
#read the datset and split into test and train
data=pd.read_csv('bmi_level.csv')
labels=pd.DataFrame(data['Index'])
features=data.drop(['Gender','Index'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=0.1,random_state=42)
clf=SVC(kernel='linear')
clf.fit(features,labels)

y_pred=clf.predict(df1)
#After prediction of obesity level we suggest the food to person
if y_pred[0] ==4 or y_pred[0] ==5:
    if calo_per_wght >2.5 :
           print("This food is of high calories so We don't recommended that")
    else :
           print("You can eat that food It is of low calories")
elif y_pred[0] ==3:
    if calo_per_wght >3.5 :
           print("This food is of high calories so We don't recommended that")
    else :
           print("You can eat that food.It is of low calories")

elif y_pred[0] ==2:
    if calo_per_wght >4.5 :
           print("This food is of high calories so We don't recommended that")
    else :
           print("You can eat that food It is of low calories")

elif y_pred[0] ==1:
    print("You can eat that food")

