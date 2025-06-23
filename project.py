import pandas as pd
import re # For regular expressions in cleaning
import seaborn as sns # Advanced statistical plotting
import matplotlib.pyplot as plt # Core plotting library
import os # For file system operations (saving plots)
import pickle # For model persistence
import streamlit as st # The main web app framework
import numpy as np # For numerical operations
from fpdf import FPDF # For generating PDF reports
st.set_page_config(page_title='🚗 AutoMetrics Dashboard', layout='wide')

# Stylish welcome header
st.markdown("""
<style>
    .big-header {
        font-size: 30px;
        font-weight: bold;
        color: #2c3e50;
    }
    .sub-header {
        font-size: 18px;
        color: #34495e;
        margin-top: -15px;
        margin-bottom: 15px;
    }
</style>
<div class='big-header'>🔍 Explore Vehicle Performance Data</div>
<div class='sub-header'>Get insights into fuel economy, model trends, and mechanical specs</div>
""", unsafe_allow_html=True)
# Core Libraries for Data Handling and Visualization


# --- Streamlit Page Configuration: Must be the very first Streamlit command! ---
# Setting up the page layout for a wide view, ideal for detailed dashboards.

# --------------------------------------------------- Data Loading & Validation Module ----------------------------------------------------
class VehicleDataLoader:
    """
    Handles loading the car dataset and validating its structure against expected columns.
    Ensures the user provides the correct data for analysis.
    """
    def __init__(self, required_columns):
        self.required_columns = required_columns # Storing the expected column names

    def load_and_verify(self, uploaded_file):
        """
        Loads the CSV file and checks if all necessary columns are present.
        Returns the DataFrame if valid, otherwise an error message.
        """
        try:
            df = pd.read_csv(uploaded_file)
            # Check if all required columns exist in the uploaded file
            if not set(self.required_columns).issubset(df.columns):
                missing_cols = set(self.required_columns) - set(df.columns)
                return f"Validation Error: Missing crucial columns in your CSV: {', '.join(missing_cols)}. Please check your file!"
            # Ensure only relevant columns are kept, and in the correct order
            df = df[self.required_columns]
            return df
        except Exception as e:
            return f"Failed to read file: {e}. Is it a valid CSV?"

# --------------------------------------------------- Data Preprocessing & Cleaning Module ----------------------------------------------------
class VehicleDataCleaner:
    """
    Applies various preprocessing steps to the vehicle dataset, including
    symbol removal, handling missing values, and stripping whitespace.
    """
    def __init__(self, dataframe):
        # Always work on a copy to prevent unintended modifications to the original dataframe upstream
        self.data_frame = dataframe.copy()

    def remove_unwanted_chars(self):
        """Removes non-alphanumeric/whitespace characters from object columns."""
        for col in self.data_frame.select_dtypes(include=['object']).columns:
            # Only keep word characters, spaces, and periods (for numbers like "2.0L")
            self.data_frame[col] = self.data_frame[col].astype(str).apply(lambda x: re.sub(r'[^\w\s\.]', '', x))
        return self.data_frame

    def manage_missing_values(self):
        """
        Handles missing values: fills numerical with mean, drops rows with missing
        critical categorical data.
        """
        # Filling numerical NaNs with column mean, a common imputation strategy
        if 'cylinders' in self.data_frame.columns:
            self.data_frame['cylinders'].fillna(self.data_frame['cylinders'].mean(), inplace=True)
        if 'displacement' in self.data_frame.columns:
            self.data_frame['displacement'].fillna(self.data_frame['displacement'].mean(), inplace=True)

        # Dropping rows where crucial categorical data might be missing or became empty after cleaning
        critical_categorical_cols = ['class', 'drive', 'fuel_type', 'make', 'model', 'transmission']
        for col in critical_categorical_cols:
            if col in self.data_frame.columns:
                # Replace empty strings or strings with only whitespace with NaN, then drop
                self.data_frame[col] = self.data_frame[col].replace(r'^\s*$', np.nan, regex=True)
                self.data_frame.dropna(subset=[col], inplace=True)
        return self.data_frame

    def trim_strings(self):
        """Removes leading/trailing whitespace from all object columns."""
        for col in self.data_frame.select_dtypes(include=['object']).columns:
            self.data_frame[col] = self.data_frame[col].astype(str).str.strip()
        return self.data_frame

    def perform_all_preprocessing(self):
        """Executes the complete preprocessing pipeline."""
        self.remove_unwanted_chars()
        self.manage_missing_values()
        self.trim_strings()
        return self.data_frame

# --------------------------------------------------- Base Class for Plotting Utilities ----------------------------------------------------
class PlottingTools:
    """
    A foundational class providing common plotting functionalities.
    Subclasses will specialize these for univariate, bivariate, etc. analyses.
    """
    def __init__(self, data_frame):
        self.df = data_frame # The DataFrame to be used for plotting

    def save_plot_to_file(self, plot_figure, output_filename):
        """Saves a matplotlib/seaborn figure to a specified file path."""
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        plot_figure.savefig(output_filename, bbox_inches='tight')
        plt.close(plot_figure) # Important: close the figure to free up memory

    def create_histogram(self, column_name, num_bins=20):
        """Generates a histogram with KDE for numerical columns."""
        fig, ax = plt.subplots()
        sns.histplot(self.df[column_name], bins=num_bins, kde=True, ax=ax)
        ax.set_title(f"Distribution of {column_name.replace('_', ' ').title()}", fontsize=14)
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        return fig

    def create_boxplot(self, x_col, y_col):
        """Generates a box plot to compare a numerical variable across categories."""
        fig, ax = plt.subplots()
        sns.boxplot(data=self.df, x=x_col, y=y_col, ax=ax)
        ax.set_title(f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()} (Box Plot)", fontsize=14)
        ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels for readability
        plt.tight_layout()
        return fig

    def create_violinplot(self, x_col, y_col):
        """Generates a violin plot for detailed distribution comparison across categories."""
        fig, ax = plt.subplots()
        sns.violinplot(data=self.df, x=x_col, y=y_col, ax=ax, inner='quartile')
        ax.set_title(f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()} (Violin Plot)", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig

    def create_countplot(self, column_name, top_n_categories=None):
        """Generates a count plot for categorical column frequencies."""
        fig, ax = plt.subplots()
        if top_n_categories:
            # Show only the top N categories for cleaner visualization
            top_cats = self.df[column_name].value_counts().nlargest(top_n_categories).index
            sns.countplot(data=self.df[self.df[column_name].isin(top_cats)], x=column_name, ax=ax, order=top_cats)
        else:
            sns.countplot(data=self.df, x=column_name, ax=ax)
        ax.set_title(f"Frequency of {column_name.replace('_', ' ').title()}", fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        return fig

    def calculate_summary_statistics(self):
        """Provides descriptive statistics for all columns in the DataFrame."""
        return self.df.describe(include='all').T

# --------------------------------------------------- Univariate Analysis Module ----------------------------------------------------
class UnivariateAnalyzer(PlottingTools):
    """
    Specialized class for performing and visualizing univariate (single-variable) analysis.
    Inherits common plotting methods from PlottingTools.
    """
    def plot_single_variable_distribution(self, column_name):
        """
        Plots the distribution for a single column: histogram for numerical,
        count plot for categorical (top 10 if many categories).
        """
        if pd.api.types.is_numeric_dtype(self.df[column_name]):
            return self.create_histogram(column_name)
        else:
            return self.create_countplot(column_name, top_n_categories=10) # Limiting categorical to top 10 for clarity

# --------------------------------------------------- Bivariate Analysis Module ----------------------------------------------------
class BivariateAnalyzer(PlottingTools):
    """
    Specialized class for performing and visualizing bivariate (two-variable) analysis.
    Inherits common plotting methods from PlottingTools.
    """
    def create_scatter_plot(self, x_variable, y_variable):
        """Generates a scatter plot to show relationship between two numerical variables."""
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df, x=x_variable, y=y_variable, ax=ax)
        ax.set_title(f'{y_variable.replace("_", " ").title()} vs {x_variable.replace("_", " ").title()}', fontsize=14)
        plt.tight_layout()
        return fig

    def compare_numeric_by_category_boxplot(self, category_col, numeric_col):
        """Wrapper for boxplot specifically for bivariate comparison."""
        return self.create_boxplot(category_col, numeric_col)

    def compare_numeric_by_category_violinplot(self, category_col, numeric_col):
        """Wrapper for violin plot specifically for bivariate comparison."""
        return self.create_violinplot(category_col, numeric_col)

# --------------------------------------------------- Multivariate Analysis Module ----------------------------------------------------
class MultivariateAnalyzer(PlottingTools):
    """
    Class for exploring relationships among multiple variables,
    primarily through a correlation heatmap.
    """
    def plot_correlation_matrix_heatmap(self):
        """Generates a correlation heatmap for all numerical columns."""
        numeric_df_only = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df_only.corr()
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjusted size for better visibility
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis", ax=ax, linewidths=.5) # Changed cmap
        ax.set_title("Correlation Heatmap of Numerical Features", fontsize=16)
        plt.tight_layout()
        return fig

# --------------------------------------------------- Dummy Model for Demonstration ----------------------------------------------------
class SimpleModel:
    """
    A placeholder for future machine learning models. Currently, it just simulates
    a 'training' process.
    """
    def __init__(self):
        pass # No complex initialization for a dummy model

    def simulate_training(self):
        """Returns a dictionary indicating a 'trained' dummy model."""
        # In a real scenario, this would involve actual model training (e.g., sklearn, tensorflow)
        return {"model_type": "Placeholder Predictor", "status": "Simulated Training Complete"}

# --------------------------------------------------- Persistence Handler for Models ----------------------------------------------------
class ModelPersistence:
    """
    Utility class for saving and loading Python objects (like trained models)
    using the pickle module.
    """
    def persist_model(self, model_object, file_path):
        """Saves a model object to a specified file using pickle."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model_object, f)
            return True
        except Exception as e:
            st.error(f"Error saving model: {e}")
            return False

    def load_persisted_model(self, file_path):
        """Loads a model object from a specified file using pickle."""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file not found at {file_path}. Has it been saved?")
            return None
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

# --------------------------------------------------- PDF Report Generation ----------------------------------------------------
class AnalysisReportPDF(FPDF):
    """
    Custom FPDF class to generate a comprehensive PDF report of the analysis.
    Includes custom headers, section titles, text, and image embedding.
    """
    def header(self):
        self.set_font("Arial", 'B', 18)
        self.set_text_color(50, 50, 150) # Dark blue for header
        self.cell(0, 15, "Automobile Data Exploration Report".encode('latin1', 'replace').decode('latin1'), ln=True, align='C')
        self.ln(10) # Line break

    def chapter_title(self, title_text):
        self.set_font("Arial", 'BU', 15) # Bold and Underlined
        self.set_text_color(80, 80, 80) # Greyish for sections
        self.cell(0, 10, title_text.encode('latin1', 'replace').decode('latin1'), ln=True)
        self.ln(5)

    def chapter_body(self, body_text):
        self.set_font("Arial", '', 11)
        self.set_text_color(0, 0, 0) # Black for body text
        self.multi_cell(0, 7, body_text.encode('latin1', 'replace').decode('latin1')) # Multi-line text
        self.ln(8)

    def add_plot_image(self, image_path, width_percentage=0.8):
        """Adds an image (plot) to the PDF, centering it."""
        if os.path.exists(image_path):
            # Calculate width based on page width and desired percentage
            page_width = self.w - 2 * self.l_margin
            image_width = page_width * width_percentage
            # Calculate x position to center the image
            x_position = (self.w - image_width) / 2
            self.image(image_path, x=x_position, w=image_width)
            self.ln(10)
        else:
            self.chapter_body(f"Note: Image not found at {image_path}. Plot could not be included.".encode('latin1', 'replace').decode('latin1'))

# --------------------------------------------------- Streamlit Application Logic ----------------------------------------------------
if __name__ == "__main__":
    st.title("🚗 Automotive Dataset Insights Explorer")
    st.markdown("Dive deep into car performance metrics and trends!")

    # Sidebar for file input and general information
    with st.sidebar:
        st.header("⚙️ Data Input & Settings")
        st.write("Upload your CSV file containing car data.")
        uploaded_data_file = st.file_uploader("Choose your vehicle data CSV", type="csv")
        st.info("Remember: The app expects specific columns for accurate analysis!")

    # Define the columns we're looking for in the uploaded CSV
    expected_car_columns = [
        'city_mpg', 'class', 'combination_mpg', 'cylinders', 'displacement',
        'drive', 'fuel_type', 'highway_mpg', 'make', 'model', 'transmission', 'year'
    ]

    # Main application flow starts after file upload
    if uploaded_data_file:
        data_loader = VehicleDataLoader(expected_car_columns)
        processed_data_result = data_loader.load_and_verify(uploaded_data_file)

        if isinstance(processed_data_result, pd.DataFrame):
            st.success("🎉 Data loaded and schema verified!")
            # Store the cleaned DataFrame in session state to persist across reruns
            if 'cleaned_car_df' not in st.session_state:
                st.session_state.cleaned_car_df = processed_data_result

            # --- Data Preprocessing ---
            cleaner = VehicleDataCleaner(st.session_state.cleaned_car_df)
            st.session_state.cleaned_car_df = cleaner.perform_all_preprocessing()
            st.markdown("---") # Visual separator

            # Use tabs for a cleaner, multi-section interface
            # Replaced tab layout with stacked expanders
# tab_overview, tab_uni, tab_bi, tab_multi, tab_model_report = st.tabs([
                

            with st.expander('📂 Data Overview', expanded=True):
                st.header("Data Overview & Cleaning Status")
                st.subheader("Raw Data Snippet")
                st.dataframe(processed_data_result.head())
                if st.checkbox("Show full raw data info (console output)", key="raw_info_car"):
                    st.text(processed_data_result.info()) # Use st.text to display info output
                st.subheader("Processed Data Snapshot")
                st.dataframe(st.session_state.cleaned_car_df.head())
                st.write("Data Types and Non-Null Counts after cleaning:")
                st.text(st.session_state.cleaned_car_df.info())

                st.download_button(
                    label="⬇️ Grab Cleaned Data CSV",
                    data=st.session_state.cleaned_car_df.to_csv(index=False).encode('utf-8'),
                    file_name="cleaned_vehicle_data.csv",
                    mime="text/csv",
                    key="download_cleaned_vehicle"
                )

            # --- Univariate Analysis Tab ---
            with st.expander('🔍 Univariate Analysis', expanded=False):
                st.header("Univariate Explorations: Individual Feature Insights")
                uni_analyzer = UnivariateAnalyzer(st.session_state.cleaned_car_df)

                st.subheader("Distribution of Car Class")
                class_dist_path = "reports/car_class_distribution.png"
                if 'class' in st.session_state.cleaned_car_df.columns:
                    class_plot = uni_analyzer.plot_single_variable_distribution('class')
                    uni_analyzer.save_plot_to_file(class_plot, class_dist_path)
                    st.pyplot(class_plot)
                else:
                    st.warning("Column 'class' not found for univariate analysis.")

                st.subheader("Fuel Type Frequencies")
                fuel_type_dist_path = "reports/car_fuel_type_distribution.png"
                if 'fuel_type' in st.session_state.cleaned_car_df.columns:
                    fuel_type_plot = uni_analyzer.plot_single_variable_distribution('fuel_type')
                    uni_analyzer.save_plot_to_file(fuel_type_plot, fuel_type_dist_path)
                    st.pyplot(fuel_type_plot)
                else:
                    st.warning("Column 'fuel_type' not found for univariate analysis.")

                st.subheader("Top 10 Car Makes by Count")
                make_dist_path = "reports/car_make_top10_distribution.png"
                if 'make' in st.session_state.cleaned_car_df.columns:
                    # Using create_countplot directly for specific top_n behavior
                    make_plot = uni_analyzer.create_countplot('make', top_n_categories=10)
                    uni_analyzer.save_plot_to_file(make_plot, make_dist_path)
                    st.pyplot(make_plot)
                else:
                    st.warning("Column 'make' not found for univariate analysis.")

                st.subheader("Cylinders Count Distribution")
                cylinders_dist_path = "reports/car_cylinders_distribution.png"
                if 'cylinders' in st.session_state.cleaned_car_df.columns:
                    cylinders_plot = uni_analyzer.plot_single_variable_distribution('cylinders')
                    uni_analyzer.save_plot_to_file(cylinders_plot, cylinders_dist_path)
                    st.pyplot(cylinders_plot)
                else:
                    st.warning("Column 'cylinders' not found for univariate analysis.")

            # --- Bivariate Analysis Tab ---
            with st.expander('🔁 Bivariate Analysis', expanded=False):
                st.header("Bivariate Analysis: Exploring Feature Relationships")
                bi_analyzer = BivariateAnalyzer(st.session_state.cleaned_car_df)

                main_target_variable = 'combination_mpg'
                numeric_features_for_bi = [
                    col for col in st.session_state.cleaned_car_df.select_dtypes(include=[np.number]).columns
                    if col not in ['year', main_target_variable] # Year might not be a direct "feature" in some contexts
                ]
                categorical_features_for_bi = [
                    col for col in st.session_state.cleaned_car_df.select_dtypes(include=['object']).columns
                ]

                st.subheader(f"Correlation with **{main_target_variable.replace('_', ' ').title()}**")
                selected_bi_feature_type = st.radio("Choose feature type for correlation:", ["Numeric", "Categorical"], key="bi_feature_type")

                if selected_bi_feature_type == "Numeric":
                    selected_bi_feature = st.selectbox("Select a numeric feature:", numeric_features_for_bi, key="selected_bi_numeric")
                    scatter_plot_path = "reports/bivariate_scatter_user_selected.png"
                    if selected_bi_feature:
                        try:
                            correlation_val = st.session_state.cleaned_car_df[selected_bi_feature].corr(st.session_state.cleaned_car_df[main_target_variable])
                            trend_desc = "positively correlated with" if correlation_val > 0 else "negatively correlated with"
                            st.info(f"Insight: **{selected_bi_feature.replace('_', ' ').title()}** is {trend_desc} **{main_target_variable.replace('_', ' ').title()}** (Correlation: `{correlation_val:.2f}`).")
                            plot_fig = bi_analyzer.create_scatter_plot(selected_bi_feature, main_target_variable)
                            bi_analyzer.save_plot_to_file(plot_fig, scatter_plot_path)
                            st.pyplot(plot_fig)
                        except Exception as e:
                            st.error(f"Could not generate plot for selected numeric features: {e}")
                else: # Categorical
                    selected_bi_feature = st.selectbox("Select a categorical feature:", categorical_features_for_bi, key="selected_bi_categorical")
                    boxplot_plot_path = "reports/bivariate_boxplot_user_selected.png"
                    if selected_bi_feature:
                        try:
                            # Display average target variable per category
                            avg_mpg_by_cat = st.session_state.cleaned_car_df.groupby(selected_bi_feature)[main_target_variable].mean().sort_values(ascending=False)
                            st.write(f"**Average {main_target_variable.replace('_', ' ').title()} by {selected_bi_feature.replace('_', ' ').title()}:**")
                            st.dataframe(avg_mpg_by_cat.head()) # Show top few categories for brevity
                            plot_fig = bi_analyzer.compare_numeric_by_category_boxplot(selected_bi_feature, main_target_variable)
                            bi_analyzer.save_plot_to_file(plot_fig, boxplot_plot_path)
                            st.pyplot(plot_fig)
                        except Exception as e:
                            st.error(f"Could not generate plot for selected categorical feature: {e}")

                st.markdown("---")
                st.subheader("Pre-defined Bivariate Views:")

                col_bi_auto1, col_bi_auto2 = st.columns(2)
                with col_bi_auto1:
                    st.write(f"##### {main_target_variable.replace('_', ' ').title()} by Car Class (Box Plot)")
                    mpg_class_box_path = "reports/mpg_by_class_boxplot.png"
                    if 'class' in st.session_state.cleaned_car_df.columns and main_target_variable in st.session_state.cleaned_car_df.columns:
                        top_5_classes = st.session_state.cleaned_car_df['class'].value_counts().nlargest(5).index
                        df_top_5_classes = st.session_state.cleaned_car_df[st.session_state.cleaned_car_df['class'].isin(top_5_classes)]
                        class_box_plot = BivariateAnalyzer(df_top_5_classes).compare_numeric_by_category_boxplot("class", main_target_variable)
                        bi_analyzer.save_plot_to_file(class_box_plot, mpg_class_box_path)
                        st.pyplot(class_box_plot)
                    else:
                        st.warning("Class or Combination MPG column missing for box plot.")

                    st.write(f"##### Displacement vs. {main_target_variable.replace('_', ' ').title()} (Scatter Plot)")
                    displacement_mpg_scatter_path = "reports/displacement_vs_mpg_scatter.png"
                    if 'displacement' in st.session_state.cleaned_car_df.columns and main_target_variable in st.session_state.cleaned_car_df.columns:
                        disp_mpg_scatter_plot = bi_analyzer.create_scatter_plot('displacement', main_target_variable)
                        bi_analyzer.save_plot_to_file(disp_mpg_scatter_plot, displacement_mpg_scatter_path)
                        st.pyplot(disp_mpg_scatter_plot)
                    else:
                        st.warning("Displacement or Combination MPG column missing for scatter plot.")

                with col_bi_auto2:
                    st.write(f"##### {main_target_variable.replace('_', ' ').title()} by Fuel Type (Violin Plot)")
                    fuel_type_violin_path = "reports/mpg_by_fuel_type_violin.png"
                    if 'fuel_type' in st.session_state.cleaned_car_df.columns and main_target_variable in st.session_state.cleaned_car_df.columns:
                        fuel_type_violin_plot = bi_analyzer.compare_numeric_by_category_violinplot("fuel_type", main_target_variable)
                        bi_analyzer.save_plot_to_file(fuel_type_violin_plot, fuel_type_violin_path)
                        st.pyplot(fuel_type_violin_plot)
                    else:
                        st.warning("Fuel Type or Combination MPG column missing for violin plot.")

                    st.write(f"##### Highway MPG vs. City MPG (Scatter Plot)")
                    highway_city_scatter_path = "reports/highway_vs_city_mpg_scatter.png"
                    if 'highway_mpg' in st.session_state.cleaned_car_df.columns and 'city_mpg' in st.session_state.cleaned_car_df.columns:
                        highway_city_scatter_plot = bi_analyzer.create_scatter_plot('highway_mpg', 'city_mpg')
                        bi_analyzer.save_plot_to_file(highway_city_scatter_plot, highway_city_scatter_path)
                        st.pyplot(highway_city_scatter_plot)
                    else:
                        st.warning("Highway MPG or City MPG column missing for scatter plot.")

            # --- Multivariate Analysis Tab ---
            with st.expander('🌐 Multivariate Heatmap', expanded=False):
                st.header("Multivariate Analysis: How Features Interconnect")
                st.markdown("Visualizing correlations across all numerical features to spot complex relationships.")
                multi_analyzer = MultivariateAnalyzer(st.session_state.cleaned_car_df)
                heatmap_plot_path = "reports/car_correlation_heatmap.png"

                if st.button("Generate & Show Correlation Heatmap", key="car_heatmap_btn"):
                    heatmap_fig = multi_analyzer.plot_correlation_matrix_heatmap()
                    multi_analyzer.save_plot_to_file(heatmap_fig, heatmap_plot_path)
                    st.pyplot(heatmap_fig)
                    st.success("Heatmap generated! Check the diagonal for self-correlation (always 1).")
                else:
                    st.info("Click the button to visualize the correlation matrix.")


            # --- Model & Report Tab ---
            with st.expander('🧠 Model & Report Generator', expanded=False):
                st.header("Dummy Model & Comprehensive Report Generation")
                st.subheader("Model Training Simulation")
                model_trainer = SimpleModel()
                trained_dummy_model = model_trainer.simulate_training()
                model_persister = ModelPersistence()

                model_save_dir = "models"
                os.makedirs(model_save_dir, exist_ok=True)
                dummy_model_filepath = os.path.join(model_save_dir, "dummy_vehicle_model.pkl")

                if model_persister.persist_model(trained_dummy_model, dummy_model_filepath):
                    st.success(f"✅ Placeholder model '`{trained_dummy_model['model_type']}`' saved to `'{dummy_model_filepath}'`.")
                    st.markdown("This model is a conceptual representation. In a real project, this would be a predictive model (e.g., for MPG prediction).")

                st.subheader("Final Analysis Report (PDF)")
                report_conclusion = ("This analysis explored various aspects of vehicle data, from individual attribute distributions to complex inter-feature relationships. "
                                     "Key insights into MPG, car classes, fuel types, and engine characteristics have been identified. "
                                     "This information is crucial for understanding vehicle performance, market segmentation, and potentially for strategic decision-making in the automotive industry.")
                st.markdown(report_conclusion)

                if st.button("📄 Create PDF Analysis Report", key="car_pdf_report_btn"):
                    pdf_reporter = AnalysisReportPDF()
                    pdf_reporter.add_page()

                    # Cover Page
                    pdf_reporter.chapter_title("Vehicle Data Analysis Report")
                    pdf_reporter.chapter_body("An in-depth exploration of automotive performance and characteristics.")
                    pdf_reporter.chapter_body(f"Date Generated: {pd.to_datetime('today').strftime('%Y-%m-%d')}")
                    if 'selected_bi_feature' in locals():
                        pdf_reporter.chapter_body(f"Highlight: Investigated relation between {selected_bi_feature.replace('_', ' ').title()} and {main_target_variable.replace('_', ' ').title()}.")
                    pdf_reporter.add_page()

                    # Executive Summary
                    pdf_reporter.chapter_title("1. Executive Summary")
                    pdf_reporter.chapter_body("This report details an exploratory data analysis (EDA) conducted on a dataset of vehicle specifications. The aim was to uncover patterns, distributions, and correlations among various car attributes, with a focus on fuel efficiency metrics (MPG).")
                    pdf_reporter.add_page()

                    # Univariate Section
                    pdf_reporter.chapter_title("2. Univariate Analysis: Single Feature Distributions")
                    pdf_reporter.chapter_body("Visualizations of individual column distributions provide foundational understanding of the dataset's characteristics.")
                    pdf_reporter.add_plot_image(class_dist_path, width_percentage=0.7)
                    pdf_reporter.add_plot_image(fuel_type_dist_path, width_percentage=0.7)
                    pdf_reporter.add_plot_image(make_dist_path, width_percentage=0.9)
                    pdf_reporter.add_plot_image(cylinders_dist_path, width_percentage=0.7)
                    pdf_reporter.add_page()

                    # Bivariate Section
                    pdf_reporter.chapter_title("3. Bivariate Analysis: Feature Relationships")
                    pdf_reporter.chapter_body(f"This section explores how different features relate to each other, especially concerning {main_target_variable.replace('_', ' ').title()}.")
                    if 'selected_bi_feature_type' in locals() and selected_bi_feature_type == "Numeric" and 'scatter_plot_path' in locals() and os.path.exists(scatter_plot_path):
                         pdf_reporter.chapter_body(f"Interactive Scatter Plot: {selected_bi_feature.replace('_', ' ').title()} vs {main_target_variable.replace('_', ' ').title()}")
                         pdf_reporter.add_plot_image(scatter_plot_path, width_percentage=0.8)
                    elif 'selected_bi_feature_type' in locals() and selected_bi_feature_type == "Categorical" and 'boxplot_plot_path' in locals() and os.path.exists(boxplot_plot_path):
                         pdf_reporter.chapter_body(f"Interactive Box Plot: {main_target_variable.replace('_', ' ').title()} by {selected_bi_feature.replace('_', ' ').title()}")
                         pdf_reporter.add_plot_image(boxplot_plot_path, width_percentage=0.8)
                    else:
                        pdf_reporter.chapter_body("No custom bivariate plot generated during session for inclusion here.")

                    pdf_reporter.chapter_body("Additional Bivariate Plots:")
                    pdf_reporter.add_plot_image(mpg_class_box_path, width_percentage=0.8)
                    pdf_reporter.add_plot_image(fuel_type_violin_path, width_percentage=0.8)
                    pdf_reporter.add_plot_image(displacement_mpg_scatter_path, width_percentage=0.8)
                    pdf_reporter.add_plot_image(highway_city_scatter_path, width_percentage=0.8)
                    pdf_reporter.add_page()

                    # Multivariate Section
                    pdf_reporter.chapter_title("4. Multivariate Analysis: Overall Correlations")
                    pdf_reporter.chapter_body("A correlation heatmap provides a holistic view of linear relationships between all numerical attributes.")
                    if os.path.exists(heatmap_plot_path):
                        pdf_reporter.add_plot_image(heatmap_plot_path, width_percentage=0.9)
                    else:
                        pdf_reporter.chapter_body("Correlation heatmap was not generated or found.")
                    pdf_reporter.add_page()

                    # Conclusion
                    pdf_reporter.chapter_title("5. Conclusion & Future Work")
                    pdf_reporter.chapter_body(report_conclusion)
                    pdf_reporter.chapter_body("Future work could involve building predictive models for MPG, or performing clustering on vehicle features to identify distinct car segments.")

                    # Save and provide download button
                    report_output_path = "reports/vehicle_eda_report.pdf"
                    pdf_reporter.output(report_output_path)
                    with open(report_output_path, "rb") as f:
                        st.download_button(
                            label="📥 Download Full Analysis PDF",
                            data=f.read(),
                            file_name="vehicle_analysis_report.pdf",
                            mime="application/pdf"
                        )
        else:
            st.error(processed_data_result) # Display the error message from data validation