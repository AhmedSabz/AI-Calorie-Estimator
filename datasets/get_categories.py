import json

with open("datasets/id2label.json", "r", encoding="utf-8") as f:
   categories = json.load(f)

   print("Number of categories:", len(categories))

   print("\nFirst 10 categories:")

   for class_id, food_name in list(categories.items())[:10]:
       print(f"{class_id}: {food_name}")

   print("\nSample mapping:")

   for class_id in ["0", "48", "66", "90"]:
      print(f"{class_id}: {categories[class_id]}")
