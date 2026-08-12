# White Sturgeon Streamlit App - Setup Instructions

This folder contains the Streamlit app for the White Sturgeon final project. 

## Files needed to run the app

Keep these files together in the same folder:

- `app.py`
- `requirements.txt`
- `sturgeon_segment_summary.csv`
- `sturgeon_priority_locations.csv`
- `sturgeon_month_summary.csv`
- `sturgeon_logistic_model.joblib`

The file `prepare_app_assets.py` is also included for reproducibility. You only need it if you want to rebuild the supporting files from the original raw CSV.

## Run the app on Windows

Open PowerShell or the VS Code terminal inside the app folder.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

Run Streamlit:

```powershell
python -m streamlit run app.py
```


## What the app does

The app has three main sections. `Monitoring Priorities` shows the high-priority locations that remained strong in both the EDA and the held-out logistic model. `Explore Locations` lets the user filter location results and view the monthly pattern. `Model Prediction` lets the user choose a waterbody-RKM segment, month, and year and see the logistic regression probability for the large mature class.

The prediction is not a catch probability. The original dataset only contains historical capture records and does not contain survey effort or unsuccessful sampling attempts. The app therefore treats the model as a prioritization tool rather than an estimate of monitoring efficiency or fish abundance.

## Rebuild the app files from the raw CSV

If you want to recreate the prepared files, put the original dataset in the app folder and rename it:

`sturgeon-pit-tag-dat.csv`

Then run:

```bash
python prepare_app_assets.py
```

This recreates the three supporting CSV files and `sturgeon_logistic_model.joblib`. After that, the raw CSV can be removed from the app folder before running or deploying the app.
