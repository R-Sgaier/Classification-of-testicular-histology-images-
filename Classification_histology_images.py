
import os
from PIL import Image
import shutil
from img2vec_pytorch import Img2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# Convert histology images from TIF to JPG
# Set the path of the directory containing the TIF files

tiff_dir = './Phenotype_Classification_Project/Tiff_dir'

# Set directory for saving JPG files
jpg_dir = './Phenotype_Classification_Project/Jpeg_dir'

# Loop through all files in the TIF directory
for filename in os.listdir(tiff_dir):
    if filename.lower().endswith('.tif') or filename.lower().endswith('.tiff'):
        # Construct full file path
        tif_path = os.path.join(tiff_dir, filename)
        # Open the TIF file
        with Image.open(tif_path) as img:
            # Convert the image to RGB
            rgb_img = img.convert("RGB")

            # Construct the output file path
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(jpg_dir, jpg_filename)

            # Save the image in JPEG format
            rgb_img.save(jpg_path, 'JPEG')

        print(f'Converted {filename} to {jpg_filename}')


# Move the JPG files to subfolders to be used in model training
# categories = different testicular phenotypes (contained in file names)

# Define the source and target folders
source_folder = jpg_dir  
target_dir = './Phenotype_Classification_Project/data_ph/Training_images'  

# Define the target folders
# HYPO = Hypospermatogenesis, NORSP = Normal Spermatogenesis, SCO = Sertoli cell only syndrome, TUBS = Tubular shadows
#Create  dictionary mapping strings contained in file names ('Hyposperm', 'Normal', 'SCO', 'Tubular') to their respective target directories
folders = {
    'Hyposperm': os.path.join(target_dir, 'HYPO'),
    'Normal': os.path.join(target_dir, 'NORSP'),
    'SCO': os.path.join(target_dir, 'SCO'),
    'Tubular': os.path.join(target_dir, 'TUBS')
}

# Loop through all files in the source directory
for filename in os.listdir(source_folder):
    if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
        # Determine the target folder based on the filename
        target_folder = None
        for key, folder in folders.items():
            if key in filename:
                target_folder = folder
                break
        
        # If a target folder was found, move the file
        if target_folder:
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            shutil.move(source_path, target_path)
            print(f'Moved {filename} to {target_folder}')
        else:
            print(f'No match found for {filename}')



# Set up directories for training and testing images

data_dir = './data_ph'
train_dir = os.path.join(data_dir, 'Training_images')
test_dir = os.path.join(data_dir, 'Testing_images')

img2vec = Img2Vec()

# Create variable to store extracted data
data = {}
# Iterate through the subdirectories, containing different categories (phenotypes)
# Extract features using Img2Vec
for j, dir_ in enumerate([train_dir, test_dir]):
    features = []
    labels = []
    for category in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_, category)):
            img_path_ = os.path.join(dir_, category, img_path)
            img = Image.open(img_path_)

            img_features = img2vec.get_vec(img)

            features.append(img_features)
            labels.append(category)

    data[['training_data', 'testing_data'][j]] = features
    data[['training_labels', 'testing_labels'][j]] = labels


# Fit and test model

model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])
pred_p = model.predict(data['testing_data'])
score = accuracy_score(pred_p, data['testing_labels'])


