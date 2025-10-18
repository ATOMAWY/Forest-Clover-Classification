# Forest Cover Type Classification

A multi-class classification project that predicts forest cover types from cartographic variables using Random Forest and XGBoost algorithms.

## 📊 Dataset
- **Source**: [Forest Cover Type (Kaggle/UCI)](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
- **Features**: 54 cartographic features (elevation, slope, distance to water, soil types, etc.)
- **Target**: 7 forest cover types
- **Samples**: 581,012 instances

## 🛠️ Technologies Used
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## 📁 Project Structure
```
├── main.py                           # Main script
├── covtype.csv                       # Dataset (download separately)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/ML-Task3-Forest-Cover-Classification.git
cd ML-Task3-Forest-Cover-Classification
```

2. **Create a virtual environment:**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download the dataset:**
- Download from [Kaggle](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset)
- Save as `covtype.csv` in the project folder

5. **Run the script:**
```bash
python main.py
```

## 📈 Features
- Data exploration and visualization
- Train-test split with stratification
- Random Forest classifier implementation
- XGBoost classifier implementation
- Model comparison and evaluation
- Feature importance analysis
- Confusion matrix visualization
- Performance metrics (accuracy, precision, recall, F1-score)

## 📊 Results
Both models achieve high accuracy in predicting forest cover types:
- Random Forest: ~95% accuracy
- XGBoost: ~95% accuracy

The feature importance analysis reveals that elevation, distance to water, and soil type are the most significant predictors.

## 📝 License
This project is open source and available under the MIT License.

## 👤 Author
Your Name - [GitHub Profile](https://github.com/YOUR_USERNAME)
