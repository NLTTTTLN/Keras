import pandas

# 1. Load dữ liệu và phân mảnh dữ liệu cho Train, Val và Test
health_data = pandas.read_csv("health_data_with_activity.csv")
health_data.head(15)
print(health_data)