# 🧪 co-fitting: Optical Density Curve Fitting Tool

**co-fitting** is an interactive Streamlit-based web application designed for analyzing and fitting optical density (OD) data, especially from microplate reader experiments. It enables users to perform background correction, visualize selected wells, and apply curve-fitting models across growth phases.

## 🚀 Features

- Upload Excel/CSV files with OD data  
- Select wells interactively from a microplate layout  
- Background correction via blank well subtraction or arithmetic operations  
- Custom model fitting with user-defined initial parameters  
- Manual phase selection and phase-wise analysis  
- Visualization of confidence intervals and standard deviations  
- Support for multiple experimental groups

## 📂 Project Structure

```
co-fitting/
│
├── app.py                 # Main Streamlit application
├── utils/                 # Helper modules for plotting, fitting, layout
├── assets/                # Contains static assets (e.g. layout images)
├── requirements.txt       # Dependency list
└── README.md              # Project documentation
```

## 📊 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/srinathlaka/co-fitting.git
cd co-fitting
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Then open the displayed URL in your browser to interact with the app.

## ⚙️ Customization Options

- Easily integrate your own growth models in `utils/fitting.py`  
- Modify layout grids (e.g., 96-well or custom) in the layout utilities  
- Adjust background correction logic and analysis methods as needed

## 👨‍💻 Author

**Srinath Laka**  
M.Sc. Scientific Instrumentation, EAH Jena  
[GitHub](https://github.com/srinathlaka)  
[LinkedIn](https://www.linkedin.com/in/srinathlaka)

## 📃 License

Licensed under the MIT License. See the `LICENSE` file for details.
