import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Connect to Firebase
cred = credentials.Certificate('firebase_key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Send a test reading
test_data = {
    'timestamp': datetime.now().isoformat(),
    'total_power': 3.45,
    'fridge': 0.15,
    'washing_machine': 2.0,
    'tv': 0.1,
    'lighting': 0.06,
    'water_heater': 3.0,
    'oven': 0.0,
    'estimated_daily_bill': 5.32
}

# Push to Firebase
db.collection('energy_readings').add(test_data)
print("✅ Data sent to Firebase successfully!")
print(f"   Sent: {test_data}")