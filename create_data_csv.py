import pandas as pd
import numpy as np

# Đặt số lượng mẫu
num_samples = 500

# Tạo dữ liệu ngẫu nhiên cho nồng độ oxy, nhịp tim và trạng thái hoạt động
np.random.seed(0)  # Để có kết quả lặp lại

# Nồng độ oxy trong máu
sp02 = np.random.uniform(85, 100, num_samples)  # Nồng độ oxy từ 85% đến 100%
sp02 = np.round(sp02).astype(int)
# Nhịp tim
bpm = np.random.uniform(60, 130, num_samples)  # Nhịp tim từ 60 đến 130 bpm
bpm = np.round(bpm).astype(int)
# Trạng thái hoạt động
activity_states = np.random.choice(['Nghỉ ngơi', 'Đi bộ', 'Hoạt động'], num_samples)


# Xác định tình trạng sức khỏe dựa trên nồng độ oxy, nhịp tim và trạng thái hoạt động
def determine_health_status(oxy, heart_rate, activity):
    if heart_rate > 120: # Nhịp tim vượt quá 120bpm
        return 'Nguy kịch'
    elif heart_rate > 100:# Nhịp tim vượt quá 100bpm
        return 'Cảnh báo'
    elif oxy < 90 and activity == 'Hoạt động':# Mức SpO2 đạt dưới 90% và người đeo đang trong trạng thái hoạt động
        return 'Cảnh báo'
    elif oxy < 85:# Mức SpO2 dưới 85%
        return 'Nguy kịch'
    else:
        return 'Bình thường'

#Tạo danh sách giá trị health_status dựa trên các tham số sp02, bpm, activity_states
health_status = [determine_health_status(o, h, a) for o, h, a in zip(sp02, bpm, activity_states)]

# Tạo DataFrame
df = pd.DataFrame({
    'ID': range(1, num_samples + 1),
    'Nồng độ Oxy (%)': sp02,
    'Nhịp Tim (bpm)': bpm,
    'Trạng Thái Hoạt Động': activity_states,
    'Tình Trạng Sức Khỏe': health_status
})

# Lưu DataFrame vào file CSV
df.to_csv('health_data_with_activity.csv', index=False)
