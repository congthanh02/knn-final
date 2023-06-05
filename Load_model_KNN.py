from preprocessing import SimplePreprocessor # Import modul SimplePreprocessor
from datasets import simpledatasetloader # Import modul simpledatasetloader
import cv2
import pickle  #Thư viện này để đọc file model
from gtts import gTTS
#import playsound
import playsound
import time

# Khởi tạo danh sách nhãn
classLabels = ["cat", "dog", "duck", "panda"]
classLabels_TV = ["mèo", "chó", "vịt", "gấu"]

print("[Thông Báo] Đang nạp ảnh để bộ phân lớp dự đoán...")

# Thiết lập kích thước ảnh 32 x 32
sp = SimplePreprocessor(32, 32)
# Tạo bộ nạp dữ liệu ảnh
sdl = simpledatasetloader.SimpleDatasetLoader(preprocessors=[sp])

# Nạp dữ liệu file ảnh và lưu dưới dạng mảng
data, _ = sdl.load(["Image\\testduck.jpg"])

#Thay đổi cách biểu diễn mảng dữ liệu ảnh
data = data.reshape((data.shape[0], 3072))

# Nạp model KNN đã train
print("[Thông Báo] Nạp model k-NN ...")
model = pickle.load(open('knn.model', 'rb'))

# Dự đoán
print("[Thông Báo] Thực hiện dự đoán ảnh để phân lớp...")

preds = model.predict(data) # Trả về danh sách nhãn dự đoán: 0->cat, 1->dog, 2->duck
print(preds)


#lưu âm thanh của tt vào file output.mp3
text = "Đây là con "  + classLabels_TV[preds[0]]
output = gTTS(text, lang="vi", slow=False)
output.save("output.mp3")

# Đọc file ảnh
image = cv2.imread("Image\\testduck.jpg")

# Thay đổi kích thước ảnh thành 500x500
resized_image = cv2.resize(image, (500, 500))

# Viết label lên ảnh
cv2.putText(resized_image, "label: {}".format(classLabels[preds[0]]), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Hiển thị ảnh
cv2.imshow("Image", resized_image)

#if classLabels =panda:



playsound.playsound('output.mp3', True)# Phát âm thanh label là con vật gì
# Chờ người dùng nhấn phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)

# Đóng cửa sổ
cv2.destroyAllWindows()


