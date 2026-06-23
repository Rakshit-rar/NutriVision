import pandas as pd

class NutriSearch:
    def __init__(self, nutrition_csv='nutrition.csv'):
        try:
            self.df = pd.read_csv(nutrition_csv)
        except:
            self.df = pd.DataFrame(columns=['food_name', 'calories', 'protein_g', 'carbs_g', 'fat_g'])

    def get_nutrition_by_name(self, food_name):
        match = self.df[self.df['food_name'].str.contains(food_name, case=False, na=False)]
        if match.empty:
            return {"error": "Data not found"}
        row = match.iloc[0]
        return {
            "food": row['food_name'],
            "calories": row['calories'],
            "protein": row['protein_g'],
            "carbs": row['carbs_g'],
            "fat": row['fat_g']
        }
