from sklearn.neighbors import KNeighborsClassifier
# LabelEncoder, một tiện ích để convert nhãn thành số nguyên
from sklearn.preprocessing import LabelEncoder
# Hàm train_test_split được sử dụng để tách dữ liệu: dữ liệu train và dữ liệu test.
from sklearn.model_selection import train_test_split
# Hàm classification_report để hiển thị kết quả đánh giá hiệu suất của bộ phân loại
from sklearn.metrics import classification_report
import search_k
# Gọi Bộ tiền xử lý dữ liệu (để chuyển kích thước ảnh về 32 x 32)
from preprocessing import SimplePreprocessor
# Gọi bộ nạp dữ liệu, khai báo tham số verbose=500 cho biết nạp 500 ảnh để xử lý 1 lần
from datasets import SimpleDatasetLoader
from imutils import paths
import pickle  #Thư viện này để làm việc với file định dạng theo pickle (File model đã huấn luyện)

# Lấy danh sách các ảnh trong folder
print("[Thông Báo] Nạp ảnh...")
imagePaths = list(paths.list_images("datasets/animals"))
# Khởi tạo bộ tiền xử lý ảnh và nạp ảnh từ foder, thay đổi kích thước ảnh
sp = SimplePreprocessor(32, 32) # Thiết lập kích thước ảnh 32 x 32
sdl = SimpleDatasetLoader(preprocessors=[sp]) # Tạo bộ nạp dữ liệu ảnh
(data, labels) = sdl.load(imagePaths, verbose=500) #Nạp dữ liệu ảnh (mỗi lần 500 ảnh) đã được gắn nhãn trong "datasets/animals"
data = data.reshape((data.shape[0], 3072)) #Thay đổi cách biểu diễn mảng dữ liệu ảnh

# Hiển thị thông tin bộ nhớ đã dùng
# Dung lượng = (32x32x3) *3000 ảnh = 3072*3000 ~ 9MB
print("[Thông Báo] Dung lượng của bộ nhớ chứa dữ liệu ảnh: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

# Chuyển nhãn từ chuỗi sang số nguyễn
le = LabelEncoder()
labels = le.fit_transform(labels)

# Chia dữ liều thành 2 phần: 80% để train và 20% để test
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.2, random_state=24)

# Tạo mô hình (Bộ phân loại k-NN)
print("[Thông Báo] Đánh giá Bộ phân lớp k-NN ...")

# k = search_k.search(trainX, trainY, testX,  testY) #23
k = 23
model = KNeighborsClassifier(n_neighbors=k)

# Huấn luyện model (bộ phân loại k-NN)
model.fit(trainX, trainY)

#Lưu file model (Bộ phân loại k-NN sau khi huấn luyện)
pickle.dump(model, open("knn.model", 'wb'))

# Đánh giá hiệu suất Bộ phân lớp k-NN và hiển thị kết quả
print(classification_report(testY, model.predict(testX),target_names=le.classes_))

# Dự đoán nhãn cho tập dữ liệu test
predictions = model.predict(testX)

# Tính độ chính xác bằng cách so sánh kết quả dự đoán với nhãn thực tế
accuracy = (predictions == testY).mean()
print("Độ chính xác trên tập dữ liệu test: {:.2f}%".format(accuracy * 100))
