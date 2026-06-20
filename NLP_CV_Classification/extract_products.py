import csv, os
import requests

def extract_products(num=10, filename="champagne_products.csv", ingredient="champagne"):
    app_id, app_key = os.environ.get("EDAMAM_APP_ID"), os.environ.get("EDAMAM_APP_KEY")
    if not app_id or not app_key:
        print("Error: Set EDAMAM_APP_ID and EDAMAM_APP_KEY")
        return
    params = {"app_id": app_id, "app_key": app_key, "ingr": ingredient, "nutrition-type": "cooking"}
    try:
        data = requests.get("https://api.edamam.com/api/food-database/v2/parser", params=params, timeout=30).json()
        products = [{"foodId": h["food"]["foodId"], "label": h["food"]["label"], "category": h["food"].get("category","N/A"), "foodContentsLabel": h["food"].get("foodContentsLabel","N/A"), "image": h["food"].get("image","N/A")} for h in data.get("hints",[])[:num]]
        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["foodId","label","category","foodContentsLabel","image"])
            w.writeheader(); w.writerows(products)
        print(f"Saved {len(products)} products to {filename}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_products()
    print("extract_products executed (needs env vars)")
