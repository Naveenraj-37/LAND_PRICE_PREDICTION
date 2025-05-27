from flask import Blueprint, render_template, request
import pandas as pd
import numpy as np
from model import LandPricePredictor  

search_bp = Blueprint('search', __name__)

# Load and prepare data
def load_data():
    df = pd.read_csv('data/land_prediction_Dataset.csv')
    df.columns = df.columns.str.strip()  # Clean column names
    return df

# Initialize the predictor and load data
predictor = LandPricePredictor('data/land_prediction_Dataset.csv')
df = load_data()

def perform_search_logic(area, land_use_type, min_price, max_price):
    # Debug print search parameters
    print(f"\nSearch Parameters - Area: '{area}', Type: '{land_use_type}', "
          f"Min Price: {min_price}, Max Price: {max_price}")
    
    # Start with all data
    filtered = df.copy()
    print(f"Initial records: {len(filtered)}")

    # Apply filters with debug
    if area and area != 'All':
        area_clean = area.strip().lower()
        filtered = filtered[
            filtered['Area'].str.strip().str.lower() == area_clean
        ]
        print(f"After area filter ('{area}'): {len(filtered)} records")
        print(f"Found areas: {filtered['Area'].unique()}")

    if land_use_type and land_use_type != 'All':
        type_clean = land_use_type.strip().lower()
        filtered = filtered[
            filtered['Land Use Type'].str.strip().str.lower() == type_clean
        ]
        print(f"After type filter ('{land_use_type}'): {len(filtered)} records")

    if min_price:
        min_price = float(min_price)
        filtered = filtered[filtered['Land Price (?/sq.ft)'] >= min_price]
        print(f"After min price ({min_price}): {len(filtered)} records")

    if max_price:
        max_price = float(max_price)
        filtered = filtered[filtered['Land Price (?/sq.ft)'] <= max_price]
        print(f"After max price ({max_price}): {len(filtered)} records")

    if filtered.empty:
        print("No results found after all filters")
        return [], [], 0, 0, True

    # Generate predictions
    predictions = []
    for _, row in filtered.iterrows():
        area_name = row['Area']
        try:
            prediction_data = predictor.predict_for_area(area_name)
            predicted_price = prediction_data['Current Price']
            predictions.append(round(predicted_price, 2))
        except Exception as e:
            print(f"Prediction failed for {area_name}: {str(e)}")
            predictions.append(0)

    # Calculate averages
    current_avg = filtered['Land Price (?/sq.ft)'].mean()
    predicted_avg = np.mean(predictions) if predictions else 0

    print(f"Returning {len(filtered)} results")
    return filtered.to_dict('records'), predictions, current_avg, predicted_avg, False

@search_bp.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'GET':
        areas = df['Area'].unique()
        land_use_types = df['Land Use Type'].unique()
        return render_template('search/search_index.html', 
                             areas=areas, 
                             land_use_types=land_use_types)
    else:
        # Process the search form
        area = request.form.get('area', '').strip()
        land_use_type = request.form.get('land_use_type', '').strip()
        min_price = request.form.get('min_price', type=float, default=0)
        max_price = request.form.get('max_price', type=float, default=float("inf"))

        results, predictions, current_avg, predicted_avg, no_results = perform_search_logic(
            area, land_use_type, min_price, max_price
        )

        return render_template('search/search_result.html', 
                            area=area, 
                            land_use_type=land_use_type, 
                            current_avg=current_avg, 
                            predicted_avg=predicted_avg, 
                            results=results, 
                            predictions=predictions, 
                            no_results=no_results,
                            zip=zip)  