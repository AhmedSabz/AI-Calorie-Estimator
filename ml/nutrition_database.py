NUTRITION_DATABASE = {
    "steak": {
        "calories_per_100g": 250,
        "protein_per_100g": 26,
    },

    "chicken": {
        "calories_per_100g": 165,
        "protein_per_100g": 31,
    },

    "rice": {
        "calories_per_100g": 130,
        "protein_per_100g": 2.7,
    },

    "potato": {
        "calories_per_100g": 87,
        "protein_per_100g": 1.9,
    },

    "bread": {
        "calories_per_100g": 265,
        "protein_per_100g": 9,
    },

    "sausage": {
        "calories_per_100g": 300,
        "protein_per_100g": 12,
    },

    "banana": {
        "calories_per_100g": 89,
        "protein_per_100g": 1.1,
    },

    "strawberry": {
        "calories_per_100g": 32,
        "protein_per_100g": 0.7,
    },

    "ice cream": {
        "calories_per_100g": 207,
        "protein_per_100g": 3.5,
    },
}


def get_nutrition(food_name):
    """
    Return nutrition information for a food.
    """

    return NUTRITION_DATABASE.get(food_name)