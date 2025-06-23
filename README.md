
# 🚗 AutoMetrics Dashboard

**AutoMetrics Dashboard** is an interactive Streamlit web app that allows users to explore, clean, and visualize automotive dataset insights. It supports univariate, bivariate, and multivariate analysis along with PDF report generation and dummy model simulation.

---

## 🔍 Features

- 📥 Upload and validate vehicle CSV datasets
- 🧹 Data cleaning (missing values, symbols, formatting)
- 📊 Visualizations (histograms, box plots, scatter plots, violin plots)
- 🌐 Correlation heatmaps for multivariate analysis
- 🧠 Dummy model training simulation
- 📄 Auto-generated PDF reports with plots and insights

---

## ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib & Seaborn
- FPDF (PDF reports)

---

## ▶️ How to Run

1. Clone this repo:
```bash
git clone https://github.com/Bilal191919/auto-metrics-dashboard.git
cd auto-metrics-dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:
```bash
streamlit run project.py
```

4. Upload your `.csv` file when prompted on the sidebar.

---

## 📝 CSV File Requirements

Your dataset must include the following columns:
- `city_mpg`, `class`, `combination_mpg`, `cylinders`, `displacement`
- `drive`, `fuel_type`, `highway_mpg`, `make`, `model`, `transmission`, `year`

---

## 📌 Output

- Cleaned data preview
- Visual plots
- Summary statistics
- Downloadable PDF analysis report

---

## 👤 Author

**Your Name**  
Student of BS Data Science  
GitHub: [@Bilal191919](https://github.com/Bilal191919)

---

## 📜 License

This project is licensed under the MIT License.
