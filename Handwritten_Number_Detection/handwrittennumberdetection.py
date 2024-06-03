from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
mnist=fetch_openml('mnist_784',version=1)
X,y=mnist["data"], mnist["target"]
y=y.astype(int)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
svm_clf=SVC(kernel='rbf',gamma='scale')
svm_clf.fit(X_train_scaled, y_train)
y_pred=svm_clf.predict(X_test_scaled)
accuracy=accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print(classification_report(y_test, y_pred))
def plot_digit(data):
    image=data.reshape(28,28)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis("off")
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plot_digit(X_test.iloc[i])
    plt.title(f"Pred: {y_pred[i]}")
plt.show()
