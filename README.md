# Heart Disease Prediction Project

This project is part of the **AI & ML Summer Camp** (Sprints x Microsoft).  
We developed a full ML pipeline to predict heart disease, from preprocessing to deployment.

---

## 🚀 Project Structure
Heart_Disease_Project/
│
├── data/ # Raw & processed datasets
├── notebooks/ # Jupyter notebooks for each step
├── models/ # Exported models (.pkl)
├── ui/ # Streamlit app (app.py)
├── deployment/ # Ngrok/pyngrok setup & scripts
├── results/ # Metrics & visualizations
├── requirements.txt # Environment dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore file

---

## 📊 Steps Implemented
1. **Data Preprocessing & Cleaning (2.1)**  
   - Missing values handled, categorical encoding, scaling  
   - EDA: histograms, correlation heatmap, boxplots  

2. **Dimensionality Reduction (2.2)**  
   - PCA applied  
   - Cumulative variance plot  
   - Scatter plot of first 2 PCs  

3. **Feature Selection (2.3)**  
   - Random Forest importance  
   - Recursive Feature Elimination (RFE)  
   - Chi-Square test  

4. **Supervised Learning (2.4)**  
   - Logistic Regression, Decision Tree, Random Forest, SVM  
   - Metrics: Accuracy, Precision, Recall, F1-score, AUC  
   - ROC Curves  

5. **Unsupervised Learning (2.5)**  
   - K-Means clustering (Elbow method)  
   - Hierarchical clustering (Dendrogram)  
   - Compared clusters with labels  

6. **Hyperparameter Tuning (2.6)**  
   - GridSearchCV & RandomizedSearchCV  
   - Tuned Random Forest chosen as best  

7. **Model Export & Deployment (2.7)**  
   - Full preprocessing + feature selection + tuned RF saved in pipeline  
   - Saved as `models/final_model.pkl`  

8. **Streamlit UI (2.8)** [Bonus]  
   - Interactive web app for user input & prediction  
   - Visualizations of trends  

9. **Ngrok Deployment (2.9)** [Bonus]  
   - Deployment instructions in `deployment/ngrok_setup.txt`  
   - `start_with_pyngrok.py` for public link  

10. **GitHub Repository (2.10)**  
    - Complete repo with datasets, notebooks, models, UI, requirements, docs  

---

## 🛠️ Setup Instructions
### 1. Create virtual environment
```bash
py -3.12 -m venv venv
venv\Scripts\activate

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
python -m streamlit run Heart_Disease_Project/ui/app.py


App runs at: http://localhost:8501

4. Deploy with ngrok

See deployment/ngrok_setup.txt for details.
use pyngrok with start_with_pyngrok.py.