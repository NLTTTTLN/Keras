import pandas as pd
import numpy as np

# Đặt số lượng mẫu
num_samples = 500

# Tạo dữ liệu ngẫu nhiên cho nhịp tim, nồng độ oxy, nhiệt độ cơ thể và trạng thái di chuyển
np.random.seed(0)  # Để có kết quả lặp lại

# Nhịp tim
bpm = np.random.uniform(60, 130, num_samples)  # Nhịp tim từ 60 đến 130 bpm
bpm = np.round(bpm).astype(int)

# Nồng độ oxy trong máu
sp02 = np.random.uniform(85, 100, num_samples)  # Nồng độ oxy từ 85% đến 100%
sp02 = np.round(sp02).astype(int)

# Nhiệt độ cơ thể
temperature = np.random.uniform(35.5, 40.0, num_samples)  # Nhiệt độ cơ thể từ 35.5°C đến 40.0°C
temperature = np.round(temperature, 1)

# Trạng thái di chuyển
activity_states = np.random.choice(['Nghỉ ngơi', 'Đi bộ', 'Hoạt động'], num_samples)

# Xác định tình trạng sức khỏe dựa trên nhịp tim, nồng độ oxy, nhiệt độ cơ thể và trạng thái di chuyển
def determine_health_status(oxy, heart_rate, temp, activity):
    if heart_rate > 120:  # Nhịp tim vượt quá 120 bpm
        return 'Nguy kịch'
    elif heart_rate > 100:  # Nhịp tim vượt quá 100 bpm
        return 'Cảnh báo'
    elif oxy < 90 and activity == 'Hoạt động':  # Mức SpO2 dưới 90% khi đang hoạt động
        return 'Cảnh báo'
    elif oxy < 85:  # Mức SpO2 dưới 85%
        return 'Nguy kịch'
    elif temp > 38.5:  # Nhiệt độ cơ thể trên 38.5°C (sốt cao)
        return 'Cảnh báo'
    elif temp > 39.0:  # Nhiệt độ cơ thể trên 39°C (sốt nguy hiểm)
        return 'Nguy kịch'
    else:
        return 'Bình thường'

# Tạo danh sách giá trị health_status dựa trên các tham số sp02, bpm, temperature, activity_states
health_status = [determine_health_status(o, h, t, a) for o, h, t, a in zip(sp02, bpm, temperature, activity_states)]

# Tạo DataFrame
df = pd.DataFrame({
    'ID': range(1, num_samples + 1),
    'Nhịp Tim (bpm)': bpm,
    'Nồng độ Oxy (%)': sp02,
    'Nhiệt độ Cơ thể (°C)': temperature,
    'Trạng Thái Di Chuyển': activity_states,
    'Tình Trạng Sức Khỏe': health_status
})

# Lưu DataFrame vào file CSV
df.to_csv('health_data_with_activity_and_temp.csv', index=False)

