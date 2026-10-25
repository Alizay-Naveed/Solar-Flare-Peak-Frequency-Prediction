# Solar Flare Peak Frequency Prediction System

A supervised machine learning regression system that predicts the daily production frequency of C-class solar flares within an active sunspot region over a subsequent 24-hour window. Space weather forecasting is critical for protecting operational satellite networks, global positioning systems (GPS), and terrestrial electrical power distribution grids from catastrophic radiation bursts.

The system is trained on the unified **UCI Solar Flare Dataset** (~1,389 active region instances) and evaluated using a rigorous **10-Fold Cross-Validation** routine across 5 distinct, alternative regression architectures.

🔗 **[Live Interactive Web Demo Link (Streamlit Cloud)]** *(https://solar-flare-peak-frequency-prediction-019.streamlit.app/)*

---

## 🛠️ Project Architecture & Notebook Workflow

The core project pipeline is implemented completely from scratch within a Jupyter Notebook following a clean, modular 12-step engineering workflow:

*   **Step 1:** Import Libraries & Log Environment Package Versions
*   **Step 2:** Load Sample Data
    *   **Step 2.0:** Merge Raw Data Repositories (`flare.data1` + `flare.data2`) into a Single Full Master CSV
    *   **Step 2.1:** Upload the Dataset (Google Colab Runtime Handling)
    *   **Step 2.2:** Read the Dataset into a Pandas DataFrame
*   **Step 3:** Understand and Pre-process Sample Data
    *   **Step 3.1:** Understand Sample Data Dimension & Target Right-Skewness
    *   **Step 3.2:** Pre-process Sample Data (Preserving Complete Feature Variant Layouts)
*   **Step 4:** Feature Extraction (Slicing All 10 Input Structural & Historical Attributes)
*   **Step 5:** Build the Pre-processing Pipeline (`ColumnTransformer` Scaling & Encoding Matrices)
*   **Step 6:** Define the 5 Alternative Regression Algorithms
*   **Step 7:** Set Up 10-Fold Cross Validation
    *   **Step 7.1:** Create the K-Fold Shuffling Splitter
    *   **Step 7.2:** Inspect the Statistical Variance of the Folds
*   **Step 8:** Run 10-Fold Cross Validation for All 5 Algorithms (with Isolated Clone Routines)
*   **Step 9:** Compare the Algorithms
    *   **Step 9.1:** RMSE of Each Algorithm on Each Fold (Winning Folds Highlighted Matrix)
    *   **Step 9.2:** Summary Comparison & Performance Ranking Table
    *   **Step 9.3:** Visual Comparison Charts (Mean vs. Dispersion Error Plots)
    *   **Step 9.4:** Documented Analytical Observations
*   **Step 10:** Train the Final Model
    *   **Step 10.1:** Select the Best Algorithm (AdaBoost Regressor)
    *   **Step 10.2:** Train the Best Algorithm on All Available Training Data
    *   **Step 10.3:** Save the Trained Model Pipeline via Pickle Serialization
*   **Step 11:** Execute the Application Phase (Streamlit Live Inference Engine Build)
*   **Conclusion**

---

## 📥 Dataset Schema Specifications (10+1 Vector Mapping)

The model leverages the complete astronomical schema from the official UCI repository data asset, utilizing 10 features representing physical geometries, development status, and short-term solar region histories:

### Input Features (X Inputs)
*   **`Code_for_class`** *(Categorical)*: Modified Zurich structural maturity classification of the active sunspot group `[A, B, C, D, E, F, H]`.
*   **`largest_spot_size`** *(Categorical)*: Geometric scale classification of the principal sunspot core `[X, R, S, A, H, K]`.
*   **`spot_distribution`** *(Categorical)*: Structural arrangement and spot density within the group interior `[X, O, I, C]`.
*   **`activity`** *(Numerical/Integer)*: Current energy flux adjustment indicators over time `[1 = reduced, 2 = unchanged]`.
*   **`evolution`** *(Numerical/Integer)*: Operational development phase of the sunspot region `[1 = decay, 2 = no growth, 3 = growth]`.
*   **`previous_24_hour_flare_activity`** *(Numerical/Integer)*: History of major recent energetic eruptions over the past day `[1 = nothing as big as an M1, 2 = one M1, 3 = more activity than one M1]`.
*   **`historically-complex`** *(Numerical/Integer)*: Boundary flag marking if the region has a twisted magnetic record `[1 = Yes, 2 = No]`.
*   **`became_complex_on_this_pass`** *(Numerical/Integer)*: Tracks if the region grew complex during its current solar disk rotation `[1 = Yes, 2 = No]`.
*   **`area`** *(Numerical/Integer)*: Total visual global surface area scale of the active solar region `[1 = small, 2 = large]`.
*   **`area_of_largest_spot`** *(Numerical/Integer)*: Specific size flag for the dominant principal core `[1 = <=5, 2 = >5]`.

### Output Target (y Target)
*   **`common_flares`** *(Numerical/Integer)*: Total production count of C-class solar flares by this region in the subsequent 24-hour window.

---

## 📊 Cross-Validation Performance Benchmarks

To meet evaluation constraints, five custom algorithms excluding classic Random Forests and standard Gradient Boosting were evaluated via **Root Mean Squared Error (RMSE)**. 

Because solar flare frequency data is heavily zero-inflated (many quiet days) and right-skewed (rare, explosive energy spikes), a logarithmic target transformation ($y_{\text{transformed}} = \log(y + 1)$) was integrated into the validation loop to compress outlier variance and stabilize scaling lines.

### Final 10-Fold Summary Evaluation Matrix

| Algorithm | Rank | Mean RMSE (Flares) | Std RMSE (Flares) | Best Fold RMSE | Worst Fold RMSE | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost Regressor** | **1** | **0.7060** | 0.1153 | 0.4805 | 0.8555 | 1.103 |
| **Ridge Regression** | **2** | **0.7172** | 0.1219 | 0.4921 | 0.8829 | 0.926 |
| **Hist Gradient Boosting** | **3** | **0.7293** | 0.1195 | 0.4957 | 0.8510 | 5.848 |
| **Support Vector Regressor (SVR)** | **4** | **0.7527** | 0.1223 | 0.5313 | 0.8981 | 2.394 |
| **Lasso Regression** | **5** | **0.7556** | 0.1259 | 0.5230 | 0.9103 | 1.113 |

### Key Insights
*   **AdaBoost Wins:** **AdaBoost Regressor achieved the lowest mean error matrix score of 0.7060 flares**, making it our Rank 1 selection. By keeping identical structural feature combinations, the boosting sequence accurately isolates volatile active regions without losing multi-flare target signals.
*   **Strong Linear Baseline:** **Ridge Regression performed exceptionally well, securing second place (0.7172 flares).** Its L2 regularization successfully handled the strong feature multicollinearity inherent to physics-based tabular inputs.
*   **SVR Underperforms:** **Support Vector Regressor (SVR) yielded an error score of 0.7527 flares**, landing in fourth place. The high-dimensional kernel structure struggled to separate subtle magnetic threshold variations from the dense background noise.
*   **Stabilized Performance:** The logarithmic target transformation compressed data skewness effectively. This resulted in an ultra-tight performance spread of less than **0.05 flares** between the best and worst algorithms across the entire alternative matrix array.

---

## 🚀 Repository Contents & Local Deployment

*   `Solar_Flare_Peak_Frequency_Prediction.ipynb`: Master Jupyter Notebook containing the full 11-step processing pipeline.
*   `solar_flare_master_full.csv`: The master, uncompressed 1,389-row integrated data asset.
*   `solar_flare_regression_model.pkl`: Serialized, universal processing model pipeline object.
*   `app.py`: Streamlit-ready web app application script with cross-version unpickling hotfixes.
*   `requirements.txt`: Locked dependency mapping configuration file.

### How to Run Locally
1. Clone this repository to your computer.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the interactive web application tool:
   ```bash
   python -m streamlit run app.py
   ```
