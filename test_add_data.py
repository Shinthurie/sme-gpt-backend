from dataset_manager import append_record_from_json

record = append_record_from_json("incoming_json/sample1.json")

print("Added record successfully:")
for key, value in record.items():
    print(f"{key}: {value}")
    