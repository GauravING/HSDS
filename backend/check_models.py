from test import models

print("\n==============================")
print("🚗 VEHICLE CLASSIFIER (vehicle_classifier.pt)")
print("==============================")
print("Task:", models["vehicle"].model.task)
print("Names:", models["vehicle"].model.names)

print("\n==============================")
print("🧍 PERSON-HEAD DETECTOR (Person_and_head_detector.pt)")
print("==============================")
print("Task:", models["person_head"].model.task)
print("Names:", models["person_head"].model.names)

print("\n==============================")
print("🪖 HELMET DETECTOR (helmet_detector.pt)")
print("==============================")
print("Task:", models["helmet"].model.task)
print("Names:", models["helmet"].model.names)

print("\n==============================")
print("🎗️ SEATBELT DETECTOR (seatbelt_detector.pt)")
print("==============================")
print("Task:", models["seatbelt"].model.task)
print("Names:", models["seatbelt"].model.names)

print("\n==============================")
print("🔢 ANPR DETECTOR (anpr_detector.pt)")
print("==============================")
print("Task:", models["anpr"].model.task)
print("Names:", models["anpr"].model.names)

print("\n==============================\n")
