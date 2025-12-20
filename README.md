# Bonus assignment for Data Bootcamp:

## Adilet Shabay

Link: https://colab.research.google.com/drive/1rGUKYEE2zOV3HonhedcxDrGR1e3dpSYY?usp=sharing

# Fine-Tuning Time Series Forecasting: PatchTST on Electricity Dataset

## Project Overview
This project focuses on fine-tuning a state-of-the-art Deep Learning model for multivariate time series forecasting. Using the Hugging Face `Trainer` API, we fine-tuned a **PatchTST** (Patch Time Series Transformer) model on the Monash Electricity dataset.

The goal was to demonstrate that a Transformer-based architecture could significantly outperform a naive persistence baseline on a large-scale dataset (>500k rows), specifically addressing the challenges of long-term dependency modeling in time series data.

## Dataset
* **Source:** Monash Time Series Forecasting Repository
* **Dataset Name:** Electricity (Hourly)
* **Scale:**
    * **Time Steps:** 26,304 hours
    * **Series (Variates):** 321
    * **Total Data Points:** ~8.4 Million (Satisfies the >500k row requirement)
* **Preprocessing:** Missing values were handled via forward/backward filling. The data was split into training, validation, and testing sets to ensure robust evaluation.

## Model Architecture & Design Discussion
**Model Selected:** `PatchTST` (Patch Time Series Transformer)

### Connection to Survey: "Deep Learning for Time Series Forecasting"
The article *"A comprehensive survey of deep learning for time series forecasting"* highlights the strengths and weaknesses of various architectures:
* **RNNs/LSTMs:** While effective for sequential data, they suffer from high computational costs and the vanishing gradient problem when modeling very long historical sequences.
* **Transformers:** The survey notes Transformers as a powerful alternative due to their **global attention mechanism**, which captures long-range dependencies effectively. However, standard Transformers suffer from $O(L^2)$ computational complexity.

**How PatchTST Addresses These Challenges:**
1.  **Patching:** Instead of processing every single time step individually (like an RNN), PatchTST segments the time series into "patches" (sub-series) which serve as input tokens. This drastically reduces the effective sequence length, solving the computational bottleneck mentioned in the survey while preserving local semantic information.
2.  **Channel Independence:** Unlike many multivariate models discussed in older surveys that mix channels (series) early on, PatchTST treats each univariate series independently. This allows the model to learn shared patterns across all 321 electricity clients without noise interference between them, leading to superior generalization.

## Performance Results
The model was trained for 3 epochs with a learning rate of 1e-4. We evaluated performance using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** on the test set.

| Metric | Baseline (Naive Persistence) | Fine-Tuned Model (PatchTST) | Improvement |
| :--- | :--- | :--- | :--- |
| **MAE** | 0.8515 | **0.3280** | **~61.5%** |
| **RMSE** | 1.1651 | **0.4813** | **~58.7%** |

* **Result:** The fine-tuned model achieved a ~60% reduction in error compared to the baseline.
* **Analysis:** The significant drop in RMSE indicates the model is highly effective at predicting volatile peaks in electricity consumption, where large errors are heavily penalized.

## Real-world usefulness (and why this is still valuable for learning)

Even though the results look strong, I do not think this specific forecasting assignment I have done is automatically meaningful in a real-world setting. In this work, I do not have important context like what each series represents, what external drivers matter, what operational constraints exist, or what the actual decision-making objective would be. In practice, forecasting electricity is meaningless unless there is specific niche reasoning for some work or data analysis. So, while the notebook shows that the model can learn patterns and beat a naive baseline, it is more practice on how to do it, rather than actually doing specific work. But in the end it was fun to play with so many tools and stumble to different mistakes, finding ways to go around it for the final solution and things to work out in the end.

