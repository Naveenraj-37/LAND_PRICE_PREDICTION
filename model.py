import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from io import BytesIO

class LandPricePredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.raw_data = pd.read_csv(data_path)
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_paths = {
            'Linear Regression': os.path.join(self.model_dir, 'linear_regression.joblib'),
            'Random Forest': os.path.join(self.model_dir, 'random_forest.joblib'),
            'KNN Regressor': os.path.join(self.model_dir, 'knn_regressor.joblib')
        }
        self.columns_path = os.path.join(self.model_dir, 'train_columns.joblib')
        self.land_types = [ 'Commercial', 'Residential', 'Industrial', 
        'Recreational', 'Agricultural', 'Forest',
        'Wetlands', 'Institutional', 'Barren']

        self.linear_model = None
        self.rf_model = None
        self.knn_model = None
        self.train_columns = None
        self.kmeans_model = None
        self.knn_similarity_model = None

        self.ensure_model_exists()
        self.load_models()

    def preprocess_data(self):
        """Prepare data for training"""
        data = self.raw_data.copy()

        # Clean column names
        data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

        # Convert categorical variables
        data = pd.get_dummies(data, columns=['area', 'land_use_type', 'development_potential'])

        # Ensure target column name is consistent
        target_col = 'land_price_(?/sq.ft)'
        if target_col not in data.columns:
            target_col = 'land_price_(?/sq.ft)'

        X = data.drop(target_col, axis=1)
        y = data[target_col]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        print("Training models...")
        X_train, X_test, y_train, y_test = self.preprocess_data()

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'KNN Regressor': KNeighborsRegressor(n_neighbors=5)
        }

        self.results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            self.results[model_name] = {
                'MSE': mse,
                'R2 Score': r2,
                'MAE': mae
            }

            joblib.dump(model, self.model_paths[model_name])
            print(f"Trained {model_name}: MSE={mse:.2f}, R2={r2:.4f}, MAE={mae:.2f}")

        joblib.dump(X_train.columns.tolist(), self.columns_path)
        print("Model training complete!")

    def ensure_model_exists(self):
        if not all(os.path.exists(path) for path in self.model_paths.values()) or not os.path.exists(self.columns_path):
            print("Models not found. Training new models...")
            self.train_model()

    def load_models(self):
        try:
            self.linear_model = joblib.load(self.model_paths['Linear Regression'])
            self.rf_model = joblib.load(self.model_paths['Random Forest'])
            self.knn_model = joblib.load(self.model_paths['KNN Regressor'])
            self.train_columns = joblib.load(self.columns_path)
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")

    def get_target_column(self):
        possible_names = ['Land Price (?/sq.ft)', 'Land Price (?/sq.ft)']
        for name in possible_names:
            if name in self.raw_data.columns:
                return name
        raise ValueError("Could not find land price column in dataset")

    def predict_for_area(self, area_name, land_type, prediction_years=12, inflation_rate=0.048):
        predictions = {
            'Current Price': 0.0,
            'Land Type': land_type,
            'Linear Regression': [],
            'Random Forest': [],
            'KNN Regressor': [],
            'Corrected Average with Inflation': []
        }

        try:
            area_name = area_name.strip().title()
            land_type = land_type.strip().title()
            
            if land_type not in self.land_types:
                raise ValueError(f"Invalid land type. Must be one of: {', '.join(self.land_types)}")
            
            if area_name not in self.raw_data['Area'].unique():
                raise ValueError(f"Area '{area_name}' not found in dataset.")

            # Filter by both area and land type
            filtered_data = self.raw_data[
                (self.raw_data['Area'] == area_name) & 
                (self.raw_data['Land Use Type'] == land_type)
            ]
            
            if filtered_data.empty:
                raise ValueError(f"No data found for area '{area_name}' with land type '{land_type}'")

            area_row = filtered_data.iloc[0]
            current_price = float(area_row[self.get_target_column()])
            predictions['Current Price'] = round(current_price, 2)

            sample_data = filtered_data.iloc[[0]].copy()
            sample_processed = pd.get_dummies(
                sample_data, 
                columns=['Area', 'Land Use Type', 'Development Potential']
            )
            sample_processed.columns = sample_processed.columns.str.lower().str.replace(' ', '_')
            
            for col in set(self.train_columns) - set(sample_processed.columns):
                sample_processed[col] = 0

            sample_processed = sample_processed[self.train_columns]

            lr_pred = float(self.linear_model.predict(sample_processed)[0])
            rf_pred = float(self.rf_model.predict(sample_processed)[0])
            knn_pred = float(self.knn_model.predict(sample_processed)[0])

            predictions['Linear Regression'] = [round(lr_pred, 2)] * prediction_years
            predictions['Random Forest'] = [round(rf_pred, 2)] * prediction_years
            predictions['KNN Regressor'] = [round(knn_pred, 2)] * prediction_years

            inflation_preds = [round(current_price, 2)]
            for _ in range(1, prediction_years):
                inflation_preds.append(round(inflation_preds[-1] * (1 + inflation_rate), 2))
            predictions['Corrected Average with Inflation'] = inflation_preds

        except Exception as e:
            logging.error(f"Prediction failed for area {area_name} and land type {land_type}: {str(e)}")
            raise ValueError(f"Could not generate predictions: {str(e)}")

        return predictions

    def plot_price_comparison(self, years, current_prices, future_prices, area_name, land_type,inflation_rate=0.048):
        try:
            years = np.array(years, dtype=int)
            current_prices = np.array(current_prices, dtype=float)
            future_prices = np.array(future_prices, dtype=float)

            current_price_array = np.full_like(years, current_prices[0], dtype=float)

            plt.figure(figsize=(12, 7))
            plt.plot(years, current_prices, 'r--o', label='Current prices')
            plt.plot(years, future_prices, 'g-^', label=f'With Inflation ({inflation_rate*100:.1f}%)')

            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Price (₹/sq.ft)', fontsize=12)
            plt.title(f'Price Trend Prediction for {area_name} ({land_type}) ({years[0]}-{years[-1]})',
                      fontsize=14, pad=20)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)

            for i, year in enumerate(years):
                if year % 2 == 0 or year == years[-1]:
                    plt.annotate(f'₹{current_prices[i]:.0f}',
                                 (year, current_prices[i]),
                                 textcoords="offset points",
                                 xytext=(0, 10),
                                 ha='center')
                    plt.annotate(f'₹{future_prices[i]:.0f}',
                                 (year, future_prices[i]),
                                 textcoords="offset points",
                                 xytext=(0, -15),
                                 ha='center')

            plt.tight_layout()

            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()

            return buf

        except Exception as e:
            print(f"Detailed plotting error: {str(e)}")
            raise

    def analyze_data(self, area_name, land_type):
        if area_name not in self.raw_data['Area'].unique():
            raise ValueError("Invalid area name! Choose from available areas in the dataset.")
        
        if land_type not in self.land_types:
            raise ValueError(f"Invalid land type. Must be one of: {', '.join(self.land_types)}")

        try:
            model = self.rf_model
            filtered_data = self.raw_data[
                (self.raw_data['Area'] == area_name) & 
                (self.raw_data['Land Use Type'] == land_type)
            ]
            
            if filtered_data.empty:
                raise ValueError(f"No data found for area '{area_name}' with land type '{land_type}'")

            area_data = filtered_data.iloc[[0]].copy()
            processed_data = pd.get_dummies(
                area_data,
                columns=['Area', 'Land Use Type', 'Development Potential']
            )
            processed_data.columns = processed_data.columns.str.lower().str.replace(' ', '_')

            missing_cols = set(self.train_columns) - set(processed_data.columns)
            for col in missing_cols:
                processed_data[col] = 0
            processed_data = processed_data[self.train_columns]

            feature_importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.train_columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)

            feature_importance_df = feature_importance_df[
                (feature_importance_df['Importance'] > 0) &
                (~feature_importance_df['Feature'].str.startswith(('area_', 'land_use_type_', 'development_potential_')))
            ]

            top_factors = feature_importance_df.head(5).to_dict('records')
            for factor in top_factors:
                if 'values' in factor and callable(factor['values']):
                    factor['factor_values'] = factor.pop('values')

            return sorted(top_factors, key=lambda x: x['Importance'], reverse=True)

        except Exception as e:
            print(f"\nError in analyze_data: {str(e)}")
            raise ValueError(f"Analysis failed: {str(e)}")

    def cluster_areas(self, land_type=None, n_clusters=5):
        """Cluster areas based on features, optionally filtered by land type"""
        if land_type and land_type not in self.land_types:
            raise ValueError(f"Invalid land type. Must be one of: {', '.join(self.land_types)}")
            
        df = self.raw_data.copy()
        if land_type:
            df = df[df['Land Use Type'] == land_type]
            
        X = pd.get_dummies(df.drop(self.get_target_column(), axis=1), drop_first=False)
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = self.kmeans_model.fit_predict(X)

        clustered_df = df.copy()
        clustered_df['Cluster'] = clusters

        print(f"\nClustering completed for {land_type if land_type else 'all'} land types:")
        for i in range(n_clusters):
            cluster_areas = clustered_df[clustered_df['Cluster'] == i]['Area'].unique()
            print(f"Cluster {i + 1}: {', '.join(cluster_areas)}")

        return clustered_df[['Area', 'Land Use Type', 'Cluster']].drop_duplicates()

    def find_similar_areas_knn(self, area_name, land_type=None, n_neighbors=5):
        try:
            df = self.raw_data.copy()
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
            
            if land_type and land_type.lower() not in df['land_use_type'].str.lower().unique():
                raise ValueError(f"Land type '{land_type}' not found in dataset")
                
            if land_type:
                df = df[df['land_use_type'].str.lower() == land_type.lower()]

            normalized_area_name = area_name.strip().lower()
            if normalized_area_name not in df['area'].str.lower().unique():
                raise ValueError(f"Area '{area_name}' not found in dataset.")

            target_col = 'land_price_(?/sq.ft)'.lower().replace(' ', '_')
            if target_col not in df.columns:
                raise ValueError("Could not find land price column in dataset")
            
            categorical_cols = ['area', 'land_use_type', 'development_potential']
            df_processed = pd.get_dummies(df, columns=categorical_cols)
            
            if target_col not in df_processed.columns:
                raise ValueError(f"Target column '{target_col}' not found in processed data")
                
            features = df_processed.drop(target_col, axis=1)

            if self.knn_similarity_model is None:
                print("\nInitializing KNN similarity model...")
                self.knn_similarity_model = KNeighborsRegressor(n_neighbors=n_neighbors+1)
                self.knn_similarity_model.fit(features, df_processed[target_col])

            area_rows = df[df['area'].str.lower() == normalized_area_name]
            if area_rows.empty:
                raise ValueError(f"No data found for area '{area_name}'.")

            area_row = area_rows.iloc[[0]]
            area_processed = pd.get_dummies(area_row, columns=categorical_cols)

            missing_cols = set(features.columns) - set(area_processed.columns)
            for col in missing_cols:
                area_processed[col] = 0
            area_processed = area_processed[features.columns]

            distances, indices = self.knn_similarity_model.kneighbors(area_processed)
            similar_indices = indices[0][1:]
            similar_areas = df.iloc[similar_indices]['area'].str.title().unique().tolist()
            
            return similar_areas[:n_neighbors]
            
        except Exception as e:
            print(f"\nError finding similar areas: {str(e)}")
            raise ValueError(f"Could not find similar areas: {str(e)}")

if __name__ == "__main__":
    predictor = LandPricePredictor('data/land_prediction_Dataset.csv')
    
    area = input("Enter the area to analyze: ").strip()
    land_type = input(f"Enter land type ({'/'.join(predictor.land_types)}): ").strip()
    
    try:
        predictions = predictor.predict_for_area(area, land_type, prediction_years=12)
        
        print(f"\nPredicted land prices for '{area.title()}' ({land_type}):")
        for model, values in predictions.items():
            if model not in ['Current Price', 'Land Type']:
                print(f"{model}: {values}")
        
        years = list(range(2025, 2037))
        
        plot_buf = predictor.plot_price_comparison(
            years,
            predictions['Random Forest'],
            predictions['Corrected Average with Inflation'],
            area.title(),
            land_type
        )
        
        with open('price_prediction.png', 'wb') as f:
            f.write(plot_buf.getbuffer())
        print("\nSaved prediction plot as 'price_prediction.png'")
        
        analysis = predictor.analyze_data(area.title(), land_type)
        print("\nKey factors influencing land price:")
        for factor in analysis:
            print(f"- {factor['Feature']}: {factor['Importance']:.4f}")
        
        print("\nArea clustering:")
        predictor.cluster_areas(land_type=land_type, n_clusters=4)
        
        similar = predictor.find_similar_areas_knn(area, land_type=land_type)
        print(f"\nAreas similar to {area.title()} ({land_type}): {', '.join(similar)}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")