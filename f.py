import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

class GUIApp:
    def __init__(self, root, data):
        self.root = root
        self.root.title("Precipitation Prediction GUI")

        self.data = data

        # GUI components
        self.label = ttk.Label(root, text="Select a day:")
        self.label.grid(row=0, column=0, padx=10, pady=10)

        self.day_var = tk.StringVar()
        self.day_combobox = ttk.Combobox(root, textvariable=self.day_var, values=list(data.index))
        self.day_combobox.grid(row=0, column=1, padx=10, pady=10)

        self.predict_button = ttk.Button(root, text="Predict", command=self.predict)
        self.predict_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.grid(row=2, column=0, columnspan=2, pady=10)

    def load_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((400, 400), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=400, height=400)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

    def predict(self):
        day_index = int(self.day_var.get())
        selected_data = self.data.loc[day_index, :].drop('PrecipitationSumInches')

        # Extract features and target variable
        X = self.data.drop(['PrecipitationSumInches'], axis=1)
        y = self.data['PrecipitationSumInches']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the precipitation for the selected day
        input_data = selected_data.values.reshape(1, -1)
        predicted_precipitation = model.predict(input_data)[0]

        # Display prediction result
        result_text = f"Predicted Precipitation: {predicted_precipitation:.2f} inches"
        print(result_text)

        # Visualization of the selected day
        plt.figure(figsize=(8, 4))
        sns.barplot(x=selected_data.index, y=selected_data.values)
        plt.title(f'Data for Day {day_index}')
        plt.xlabel('Attributes')
        plt.ylabel('Values')
        plt.show()

if __name__ == "__main__":
    # Read the cleaned data
    data = pd.read_csv("austin_final.csv")
    data.set_index('Unnamed: 0', inplace=True)  # Assuming the first column is the index

    root = tk.Tk()
    app = GUIApp(root, data)
    root.mainloop()
