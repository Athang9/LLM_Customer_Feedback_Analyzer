import pandas as pd

# Step 1: Load CSV
df = pd.read_csv("../data/Reviews.csv")

# Step 2: Drop duplicates and nulls
df.drop_duplicates(subset="Text", inplace=True)
df.dropna(subset=["Text"], inplace=True)

# Step 3: Optional – Filter by star rating (e.g. only 1–2 star reviews)
df = df[df["Score"].isin([1, 2, 5])]  # Mix of complaints + happy users

# Step 4: Shorten to 300–500 reviews for development
df_sample = df.sample(n=500, random_state=42).reset_index(drop=True)

# Step 5: Save for the app
df_sample.to_csv("../data/cleaned_reviews.csv", index=False)

print("Cleaned 500 reviews and saved as cleaned_reviews.csv")
