# Bonus assignment for Data Bootcamp:

I fine-tuned a pretrained PatchTST time-series forecasting model with Hugging Face Transformers (Trainer) on the Monash “electricity_hourly_dataset” loaded via sktime

Link: https://colab.research.google.com/drive/1rGUKYEE2zOV3HonhedcxDrGR1e3dpSYY?usp=sharing

## Dataset and loading

I used the Monash Forecasting Repository dataset called “electricity_hourly_dataset,” loaded through sktime.datasets.ForecastingData. Since the dataset is a panel (many electricity series over time), I converted it into a wide format where each column is a different series and each row is an hourly timestamp. To keep the run manageable in Colab while still staying well above 500k observations, I kept a subset of the series, but kept the total number of observations above the necessary threshold.

## Preprocessing and how the training data was created

Because forecasting models need fixed-length inputs, I turned the continuous time series into supervised training examples using sliding windows. For each example, the model sees a “context window” of past values (the history) and learns to predict the next “forecast horizon” (the future). I also standardized the data using mean and standard deviation. This was my first time doing a full time-series windowing setup like this, and it honestly helped me understand how forecasting is basically “classification-style training,” except the labels are future sequences instead of class IDs.

## Model and training approach

I fine-tuned a pretrained PatchTST forecasting model using Hugging Face Transformers and the Trainer API. PatchTST is a transformer-based approach for forecasting that uses the idea of splitting the time series into patches, which helps it handle long histories more efficiently than treating each time step separately. 

## Evaluation and baseline comparison

After training, I evaluated the model on a held-out test split using MAE and RMSE. To make the results meaningful, I also compared against a simple baseline forecast called persistence, where the prediction just repeats the last observed value across the whole forecast horizon. This baseline is like the “majority class” baseline in classification: it is simple, but it gives a clear reference point for whether the model is actually learning something.

## Results

The fine-tuned PatchTST model clearly outperformed the baseline on the test set. In standardized units (because the data was z-scored using the training set statistics), the fine-tuned model achieved MAE around 0.328 and RMSE around 0.481. The persistence baseline was much worse, with MAE around 0.851 and RMSE around 1.165. Seeing such a big gap was a nice moment because it confirmed that the fine-tuning was doing real work and the model was not just copying the last value.

## Real-world usefulness (and why this is still valuable for learning)

Even though the results look strong, I do not think this specific forecasting assignment I have done is automatically meaningful in a real-world setting. In this work, I do not have important context like what each series represents, what external drivers matter, what operational constraints exist, or what the actual decision-making objective would be. In practice, forecasting electricity is meaningless unless there is specific niche reasoning for some work or data analysis. So, while the notebook shows that the model can learn patterns and beat a naive baseline, it is more practice on how to do it, rather than actually doing specific work. But in the end it was fun to play with so many tools and stumble to different mistakes, finding ways to go around it for the final solution and things to work out in the end.
