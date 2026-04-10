from document_pipeline import save_uploaded_file, standardize_to_image, preprocess_image

path = input("Enter PDF or image path: ").strip()

raw_file = save_uploaded_file(path)
print("Saved uploaded file:", raw_file)

orig_img = standardize_to_image(raw_file)
print("Standardized image:", orig_img)

versions = preprocess_image(orig_img)
print("Preprocessed versions:")
for key, value in versions.items():
    print(f"{key}: {value}")