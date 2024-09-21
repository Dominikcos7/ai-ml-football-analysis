from ultralytics import YOLO

model = YOLO('models/best_colab_m.pt.pt')

results = model.predict('input_videos/psv.mp4', save=True)
print(results[0])
for box in results[0].boxes:
    print(box)
    