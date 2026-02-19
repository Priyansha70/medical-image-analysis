\# 🩺 Chest X-ray Pneumonia Classifier  

ResNet18 + Grad-CAM + Streamlit



An end-to-end medical image analysis project that classifies Chest X-rays as \*\*NORMAL\*\* vs \*\*PNEUMONIA\*\* using transfer learning (ResNet18) and explains predictions with \*\*Grad-CAM\*\* visualizations.



⚠️ Educational/demo project only — not medical advice.



---



\## 🔍 Project Overview



This project demonstrates:



\- Transfer learning with pretrained ResNet18

\- Fine-tuning strategy (frozen head → full fine-tune)

\- Evaluation using Accuracy, Precision, Recall, F1, ROC-AUC

\- Confusion matrix analysis

\- Grad-CAM visual explainability

\- Interactive deployment with Streamlit



---



\## 📊 Test Set Performance



\- \*\*Accuracy:\*\* 0.912  

\- \*\*F1 Score:\*\* 0.933  

\- \*\*ROC-AUC:\*\* 0.973  

\- \*\*Recall (Sensitivity):\*\* 0.985  



Confusion Matrix:

\[\[185, 49],

\[ 6, 384]]

\## 🗂 Dataset



Kaggle: \*Chest X-Ray Images (Pneumonia)\*  

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia



Expected folder structure:



data/chest\_xray/

train/

NORMAL/

PNEUMONIA/

test/

NORMAL/

PNEUMONIA/

val/ (optional)

NORMAL/

PNEUMONIA/





---



\## ⚙️ Installation



```bash

python -m venv .venv

\# Windows:

.\\.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

🏋️ Training

python -m src.train --data\_dir data/chest\_xray --out\_dir outputs --epochs\_head 2 --epochs\_finetune 3 --batch\_size 16

Saved artifacts:



outputs/best.pt



outputs/metrics.json



outputs/classes.json



🔬 Grad-CAM Test

python gradcam\_test.py

Generates:



outputs/gradcam\_test.png

🚀 Run Web App

streamlit run app/streamlit\_app.py

Upload an X-ray → get prediction + Grad-CAM heatmap.



📌 Technical Highlights

Transfer learning with torchvision ResNet18



Proper validation + test evaluation



Threshold-based classification



Visual interpretability with backward hooks



Modular project structure (src/, app/, outputs/)



Clean separation of training and inference



⚠️ Limitations

Dataset bias and acquisition artifacts may affect generalization.



High dataset performance does not imply clinical readiness.



Grad-CAM highlights correlation, not causation.



👩‍💻 Author

Priyansha Aggarwal

Computer Science \& Mathematics

University of Alberta





