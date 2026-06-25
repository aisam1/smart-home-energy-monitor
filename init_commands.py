import firebase_admin
from firebase_admin import credentials, firestore

if not firebase_admin._apps:
    cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Set all appliances to ON by default
db.collection('device_commands').document('current').set({
    'fridge':          True,
    'water_heater':    True,
    'washing_machine': True,
    'oven':            True,
    'tv':              True,
    'lighting':        True,
})
print("Device commands initialized!")