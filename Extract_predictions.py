
from img2vec_pytorch import Img2Vec
from PIL import Image
import pandas as pd
import pickle
import os

img2vec = Img2Vec()


# Load classifying model
with open('./h_class.p', 'rb') as f:
    h_class = pickle.load(f)



# Path to the directory containing the JPEG files
folder_path = './IHC_results/Images'

# Lists to store file names, number of pixels, and scores
file_names = []
pred_phenotype = []

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # Construct full file path
        jpg_path = os.path.join(folder_path, filename)

        # Open the JPEG file
        with Image.open(jpg_path) as img:
            # Extract features with img2vec
            img_features = img2vec.get_vec(img)

            # Use classifier to assign phenotype
            pred_ph = h_class.predict([img_features])

            # Append the data to the lists
            file_names.append(filename)
            pred_phenotype.append(pred_ph)


# Create a DataFrame with the collected data
data = {
    'File': file_names,
    'Predicted Phenotype': pred_phenotype,
}
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Save the Data frame to a CSV file
output_csv_path = './IHC_results/Phenotype_predictions.csv' 
df.to_csv(output_csv_path, index=False)
