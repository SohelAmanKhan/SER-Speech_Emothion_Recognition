import tkinter as tk
import os
from tkinter.filedialog import askopenfilename
def open_file():
    file =askopenfilename(filetypes =[('Python Files', '*.wav')])
    if file is not None:
        content = file
    else:
        exit
    import librosa
    import dtale as dt
    import os, glob
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    emo1={ '01':'neutral','02':'calm','03':'happy','04':'sad','05':'angry','06':'fearful','07':'disgust','08':'surprised'}#emotion by dataset
    em=['happy','sad','angry']#emotion that i need
    emo2=0
    x=[]
    y=[]
    s=pd.DataFrame()
    for f in glob.glob("C:\\Users\\David Khan\\Desktop\\New folder\\Actor_*\\*.wav"):#path of data set
        file_name=os.path.basename(f)
        emo2=emo1[file_name.split("-")[2]]
        if emo2 not in em:
            continue
        SO,s_rate =librosa.load(f)
        stft=np.abs(librosa.stft(SO))
        feature_extracted=np.array([])
        mf=np.mean(librosa.feature.mfcc(y=SO, sr=s_rate).T, axis=0)#Mel Frequency Cepstral Coefficients
        feature_extracted=np.hstack((feature_extracted, mf))
        c=np.mean(librosa.feature.chroma_stft(S=stft, sr=s_rate).T,axis=0)#chromagram
        feature=np.hstack((feature_extracted, c))   
        m=np.mean(librosa.feature.melspectrogram(SO, sr=s_rate).T,axis=0)#melspectrogram
        feature_extracted=np.hstack((feature_extracted, m))
        x.append(feature_extracted)
        y.append(emo2)    
    x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3)
    names = ["Nearest_Neighbors:", "Linear_SVM:", "Polynomial_SVM:", "RBF_SVM:", 
         "Gradient_Boosting:", "Decision_Tree:","Random_Forest:", "Naive_Bayes:"]

    classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(kernel="poly", degree=3, C=0.025),
    SVC(kernel="rbf", C=1, gamma=2),
    GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=100),
    GaussianNB()]
    scores = []
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        score = clf.score(x_test, y_test)
        scores.append([name,score])
    xy=[]
    df = pd.DataFrame()
    df['name'] = names
    df['score'] = scores
    clf=RandomForestClassifier(max_depth=5, n_estimators=100)
    clf.fit(x_train, y_train)
    SO,s_rate =librosa.load(content)
    stft=np.abs(librosa.stft(SO))
    feature_extracted=np.array([])
    mf=np.mean(librosa.feature.mfcc(y=SO, sr=s_rate).T, axis=0)#Mel Frequency Cepstral Coefficients
    feature_extracted=np.hstack((feature_extracted, mf))
    c=np.mean(librosa.feature.chroma_stft(S=stft, sr=s_rate).T,axis=0)#chromagram
    feature=np.hstack((feature_extracted, c))   
    m=np.mean(librosa.feature.melspectrogram(SO, sr=s_rate).T,axis=0)#melspectrogram
    feature_extracted=np.hstack((feature_extracted, m))
    xy.append(feature_extracted)   
    pred=clf.predict(xy)
    df.loc[len(df.index)]=['Entered File',pred]
    l= tk.Text(root,height=80,width=100,bg='#80c1ff',bd=10)
    l.pack()
    l.insert(tk.END,str(df))
    
HEIGHT = 500
WIDTH = 600
root = tk.Tk()
canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()
background_image = tk.PhotoImage(file=r'C:\Users\David Khan\Desktop\ppp.png')
background_label = tk.Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)
frame = tk.Frame(root, bg='#80c1ff', bd=5)
frame.place(relx=0.5, rely=0.1, relwidth=0.75, relheight=0.1, anchor='n')
button = tk.Button(frame, text="File", font=40, command=open_file)
button.place(relx=0.7, relheight=1, relwidth=0.3)
root.mainloop()
