import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk

def select_file(file_entry):
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("All Files", "*.*"),  
            ("Text Files", "*.txt"),  
            ("CSV Files", "*.csv"),  
            ("Images", "*.png;*.jpg;*.jpeg")
        ]
    )
    if file_path:
        file_entry.delete(0, END)  
        file_entry.insert(0, file_path)

###Do you want to edit the file? 
def edit(file_path):
    fram = Tk()
    df = pd.read_csv(file_path)
    fram.title("Product Demand Forecasting Edit File")
    fram.geometry("500x400")
    fram.configure(bg="#f0f0f0")  

    fram.grid_rowconfigure(7, weight=1)
    fram.grid_columnconfigure(1, weight=1)

    # ✅ إدخال اسم العمود للحذف
    column_name = Label(fram, text="Enter column to drop:", font=("Arial", 12, "bold"), bg="#f0f0f0")
    column_name.grid(row=0, column=0, padx=20, pady=10, sticky="w")

    column = Entry(fram, width=40, font=("Times New Roman", 12), bg="white", relief=SOLID)
    column.grid(row=0, column=1, padx=20, pady=10, sticky="ew")

    # ✅ إدخال اسم الملف الجديد للحفظ
    file_name_label = Label(fram, text="Save As (file name):", font=("Arial", 12, "bold"), bg="#f0f0f0")
    file_name_label.grid(row=1, column=0, padx=20, pady=10, sticky="w")

    file_name_entry = Entry(fram, width=40, font=("Times New Roman", 12), bg="white", relief=SOLID)
    file_name_entry.grid(row=1, column=1, padx=20, pady=10, sticky="ew")

    def drob():
        column_name_value = column.get()
        if column_name_value in df.columns:
            df.drop(columns=[column_name_value], inplace=True)
            messagebox.showinfo("Success", f"Column '{column_name_value}' has been removed.")  # ✅ إصلاح الخطأ هنا
        else:
            messagebox.showwarning("Warning", "Column not found!")  # ✅ إصلاح الخطأ هنا


    def report():
        result_window = Toplevel()
        result_window.title("Data after modification")
        result_window.geometry("900x500")
        result_window.configure(bg="white")

        result_window.grid_rowconfigure(1, weight=1)
        result_window.grid_columnconfigure(0, weight=1)

        Label(result_window, text="Data Preview", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

        frame = Frame(result_window, bg="white")
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        tree_scroll_y = Scrollbar(frame, orient=VERTICAL)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")

        tree_scroll_x = Scrollbar(result_window, orient=HORIZONTAL)
        tree_scroll_x.grid(row=2, column=0, columnspan=2, sticky="ew")

        tree = ttk.Treeview(frame, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, stretch=NO)

        for index, row in df.head(20).iterrows():
            tree.insert("", "end", values=list(row))

    def save_as_new_file():
        file_name = file_name_entry.get()
        if file_name.strip():
            try:
                df.to_csv(f"{file_name}.csv", index=False)
                messagebox.showinfo("Success", f"Data saved successfully as '{file_name}.csv'")  # ✅ إصلاح الخطأ هنا
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save data: {e}")  # ✅ إصلاح الخطأ هنا
        else:
            messagebox.showwarning("Warning", "Please enter a valid file name!")  # ✅ إصلاح الخطأ هنا

    drop_data = Button(fram, text="Drop Column", command=drob, bg="red", fg="white", font=("Arial", 12, "bold"), relief=RAISED)
    drop_data.grid(row=2, column=1, padx=20, pady=10, sticky="ew")

    show_data = Button(fram, text="Show Data", command=report, bg="blue", fg="white", font=("Arial", 12, "bold"), relief=RAISED)
    show_data.grid(row=3, column=1, padx=20, pady=10, sticky="ew")

    save_as_button = Button(fram, text="Save As", command=save_as_new_file, bg="orange", fg="white", font=("Arial", 12, "bold"), relief=RAISED)
    save_as_button.grid(row=4, column=1, padx=20, pady=10, sticky="ew")

    fram.mainloop()

### KNN Model ###
def NNK(file_path, column_name):
    df = pd.read_csv(file_path)
    ### Splitting Data ###
    x = df.drop(column_name, axis=1)
    y = df[column_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    ### Handle Non-Numeric Data ###
    # Convert categorical variables if necessary
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    # Align train & test columns (handle missing dummies)
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)
    ### Scaling Features ###
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)  
    x_test = scaler.transform(x_test)        
    ### KNN Model ###
    model_kn = KNeighborsClassifier(n_neighbors=3)
    model_kn.fit(x_train, y_train)  
    y_pred_kn = model_kn.predict(x_test)
    report = classification_report(y_test, y_pred_kn)

      # إنشاء نافذة جديدة
    result_window = Toplevel()
    result_window.title("KNN Classification Report")
    result_window.geometry("600x600")

    result_Label=Label(result_window, text="Classification Report", font=("Arial", 14, "bold"))
    result_Label.pack(pady=10)
    
    frame = Frame(result_window)
    frame.pack(expand=True, fill=BOTH)

    # 🔹 إنشاء `Scrollbar` وربطه بـ `Text`
    scrollbar = Scrollbar(frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    text_widget = Text(frame, wrap=WORD, font=("Courier", 12), yscrollcommand=scrollbar.set)
    text_widget.insert(END, report)  # إدراج التقرير داخل `Text`
    text_widget.pack(expand=True, fill=BOTH)

    scrollbar.config(command=text_widget.yview)
    def pt(x_data, y_pred):
        plt.figure(figsize=(8, 5))
        plt.plot(x_data[:, 0], x_data[:, 1], color="red", linestyle="-", marker="v")
        plt.title("Predicted Data")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()
    def savem_model(model):
        modele = Toplevel()
        modele.title("Save Model")
        modele.geometry("300x200")
        modele.configure(bg='#FEF3E2')

        text_size = ("Times New Roman", 14, "italic")

        model_Label = Label(modele, text="Enter model name:", font=text_size, bg="#FA4032", fg="white")
        model_Label.pack(pady=10)

        model_name = Entry(modele, width=30, font=("Times New Roman", 12, "italic"), bg="#FA812F")
        model_name.pack(pady=5)

        def save():
            name = model_name.get().strip()
            if name == "":
                messagebox.showerror("Error", "Model name cannot be empty!")  
                return

            try:
                joblib.dump(model, f"{name}.pkl") 
                messagebox.showinfo("Success", f"Model saved as {name}.pkl") 
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{e}") 

        save_btn = Button(modele, text="Save Model", command=save, bg="blue", fg="white", font=("Arial", 12, "bold"))
        save_btn.pack(pady=10)

        modele.mainloop()

    ### أزرار التحكم ###
    btn_plot = Button(result_window, text="Plot Data", command=lambda: pt(x_test, y_pred_kn), bg="blue", fg="white")
    btn_plot.pack(pady=10)

    save_btn = Button(result_window, text="Save Model", command=lambda: savem_model(model_kn), bg="blue", fg="white")
    save_btn.pack(pady=10)

### RandomForestClassifier Model ###
def dei(file_path, column_name):
    df = pd.read_csv(file_path)

    ### Splitting Data ###
    x = df.drop(columns=[column_name])
    y = df[column_name]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    ### Handle Non-Numeric Data ###
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)
    x_train, x_test = x_train.align(x_test, join='left', axis=1, fill_value=0)

    ### Scaling Features ###
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    ### RandomForestClassifier Model ###
    model = RandomForestClassifier(n_estimators=100,random_state=42)
    model.fit(x_train, y_train)
    y_pred_kn = model.predict(x_test)
    report = classification_report(y_test, y_pred_kn)

    ### إنشاء نافذة جديدة ###
    result_window = Toplevel()
    result_window.title("DecisionTreeClassifier")
    result_window.geometry("600x600")

    result_Label = Label(result_window, text="DecisionTreeClassifier", font=("Arial", 14, "bold"))
    result_Label.pack(pady=10)

    frame = Frame(result_window)
    frame.pack(expand=True, fill=BOTH)

    # 🔹 إنشاء `Scrollbar` وربطه بـ `Text`
    scrollbar = Scrollbar(frame)
    scrollbar.pack(side=RIGHT, fill=Y)

    text_widget = Text(frame, wrap=WORD, font=("Courier", 12), yscrollcommand=scrollbar.set)
    text_widget.insert(END, report)  # إدراج التقرير داخل `Text`
    text_widget.pack(expand=True, fill=BOTH)

    scrollbar.config(command=text_widget.yview)

    ###  دالة رسم البيانات ###
    def pt(x_data, y_pred):
        plt.figure(figsize=(8, 5))
        plt.plot(x_data, y_pred, linestyle='-', marker='o', color='red', markersize=5, label="Predicted Data")
        plt.title("Predicted Data")
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Value")
        plt.grid(True)
        plt.show()

    ###   حفظ النموذج ###
    def savem_model(model):
        modele = Toplevel()
        modele.title("Save Model")
        modele.geometry("300x200")
        modele.configure(bg='#FEF3E2')

        text_size = ("Times New Roman", 14, "italic")

        model_Label = Label(modele, text="Enter model name:", font=text_size, bg="#FA4032", fg="white")
        model_Label.pack(pady=10)

        model_name = Entry(modele, width=30, font=("Times New Roman", 12, "italic"), bg="#FA812F")
        model_name.pack(pady=5)

        def save():
            name = model_name.get().strip()
            if name == "":
                messagebox.showerror("Error", "Model name cannot be empty!")  
                return

            try:
                joblib.dump(model, f"{name}.pkl") 
                messagebox.showinfo("Success", f"Model saved as {name}.pkl") 
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model:\n{e}") 

        save_btn = Button(modele, text="Save Model", command=save, bg="blue", fg="white", font=("Arial", 12, "bold"))
        save_btn.pack(pady=10)

        modele.mainloop()

    ### أزرار التحكم ###
    btn_plot = Button(result_window, text="Plot Data", command=lambda: pt(x_test, y_pred_kn), bg="blue", fg="white")
    btn_plot.pack(pady=10)

    save_btn = Button(result_window, text="Save Model", command=lambda: savem_model(model), bg="blue", fg="white")
    save_btn.pack(pady=10)

def report(fiel_path):
        result_window = Toplevel()
        result_window.title("Data after modification")
        result_window.geometry("900x500")
        result_window.configure(bg="white")

        result_window.grid_rowconfigure(1, weight=1)
        result_window.grid_columnconfigure(0, weight=1)
        df = pd.read_csv(fiel_path)
        Label(result_window, text="Data Preview", font=("Arial", 14, "bold"), bg="white").grid(row=0, column=0, columnspan=2, pady=10)

        frame = Frame(result_window, bg="white")
        frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        tree_scroll_y = Scrollbar(frame, orient=VERTICAL)
        tree_scroll_y.grid(row=0, column=1, sticky="ns")

        tree_scroll_x = Scrollbar(result_window, orient=HORIZONTAL)
        tree_scroll_x.grid(row=2, column=0, columnspan=2, sticky="ew")

        tree = ttk.Treeview(frame, yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)
        tree.grid(row=0, column=0, sticky="nsew")

        tree_scroll_y.config(command=tree.yview)
        tree_scroll_x.config(command=tree.xview)

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        tree["columns"] = list(df.columns)
        tree["show"] = "headings"

        for col in df.columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, stretch=NO)

        for index, row in df.head(20).iterrows():
            tree.insert("","end", values=list(row))

###test window
def test(): 
    fram = Toplevel()
    fram.title("Product Demand Forecasting")
    icon = Image.open("C:/Users/HexCode/Desktop/py/learn/io.jpg")
    icon = ImageTk.PhotoImage(icon)
    fram.iconphoto(True, icon)
    fram.geometry('500x500')
    fram.configure(bg='#FEF3E2')

    text_size = ("Times New Roman", 20, "italic")

    # عنوان الترحيب
    ledel = Label(fram, text="Welcome", font=text_size, bg="#FA4032")
    ledel.grid(row=0, column=1, padx=(110,20), pady=5)  

    # إدخال مسار الملف
    ledel_file = Label(fram, text="Enter your file path or select", font=text_size, bg="#FA4032")
    ledel_file.grid(row=1, column=1, padx=(80,10), pady=5)  

    file_entry = Entry(fram, width=50, font=("Times New Roman", 12, "italic"), bg="#FA812F")
    file_entry.grid(row=2, column=1, padx=(80,10), pady=10)  
    # زر اختيار الملف
    btn_browse = Button(fram, text="select", command=lambda: select_file(file_entry), bg="#FA812F", fg="white", font=("Arial", 12, "bold"))
    btn_browse.grid(row=3, column=1, padx=(70,10), pady=10) 

    # إدخال العمود المستهدف
    Target_column = Label(fram, text="Enter your Target column", font=text_size, bg="#FA812F")
    Target_column.grid(row=4, column=1, padx=(80,10), pady=5)  

    column = Entry(fram, width=50, font=("Times New Roman", 12, "italic"), bg="#FA812F")
    column.grid(row=5, column=1, padx=(70,10), pady=10)  

    ###NNk Button
    btn_nnk = Button(fram, text="KNeighborsClassifier", command=lambda: NNK(file_entry.get(), column.get()), bg="#FA4032", fg="white", font=("Arial", 12, "bold"))
    btn_nnk.grid(row=6, column=1, padx=(70,10), pady=10)

    ###DecisionTreeClassifier Button
    btn_dei = Button(fram, text="DecisionTreeClassifier", command=lambda: dei(file_entry.get(), column.get()), bg="#FA4032", fg="white", font=("Arial", 12, "bold"))
    btn_dei.grid(row=7, column=1, padx=(70,10), pady=10)

    show_data = Button(fram, text="Show Data", command=lambda: report(file_entry.get()), bg="blue", fg="white", font=("Arial", 12, "bold"), relief=RAISED)
    show_data.grid(row=8, column=1, padx=(70,10), pady=10)

    btn_edit = Button(fram, text="edit the file", command=lambda: edit(file_entry.get()), bg="#FA4032", fg="white", font=("Arial", 12, "bold"))
    btn_edit.grid(row=9, column=1, padx=(70,10), pady=10)
    ###extra
    fram.mainloop()

###train window
# def train():

def main():
    main=Tk()
    main.title("gj")
    main.geometry('300x200')
    icon = Image.open("C:/Users/HexCode/Desktop/py/learn/io.jpg")
    icon = ImageTk.PhotoImage(icon)
    main.iconphoto(True, icon)
    main.configure(bg='#FEF3E2')
    le1=Label(main, text="test window", font=("Times New Roman", 12, "italic"), bg="#FA812F")
    le1.grid(row=1, column=1, padx=(80,10), pady=10)
    btn_test=Button(main,text="open test window",command=test, bg="blue", fg="white")
    btn_test.grid(row=2, column=1, padx=(80,10), pady=10)

    le2=Label(main, text="train window", font=("Times New Roman", 12, "italic"), bg="#FA812F")
    le2.grid(row=3, column=1, padx=(80,10), pady=10)
    btn_train=Button(main,text="open train window",command=test, bg="blue", fg="white")
    btn_train.grid(row=4, column=1, padx=(80,10), pady=10)
    main.mainloop()

main()
