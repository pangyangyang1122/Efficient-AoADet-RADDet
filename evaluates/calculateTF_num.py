import pandas as pd

# 读取数据
file_path = '/media/sqpang/新加卷/rada_target_detection/code/Finalmodel_LinearFFT+Detection/logs/Para-init_A_FFT_flip_No_attention_FPN_RD_16_32_liner_RD_cartesian/results_TrueFalse/RD-Cartesian Boxes GT TF_b=6_cft=0.6IOU=0.3.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path)

# 数据总量
total_data = len(df)

# 统计 TP、MD、FA 的数量
TP_count = len(df[df['TF'] == 'TP'])
MD_count = len(df[df['TF'] == 'MD'])
FA_count = len(df[df['TF'] == 'FA'])

# 统计 FP 分类：定位错误和分类错误
FP_data = df[df['TF'] == 'FP']

# 定位错误：类别预测正确，但 IoU < 0.5
loc_error_count = len(FP_data[(FP_data['IoU'] < 0.3)])

# 分类错误：IoU > 0.5，但类别预测错误
cls_error_count = len(FP_data[(FP_data['IoU'] >= 0.3)])

# 输出统计结果
print(f"总数据量: {total_data}")
print(f"TP (True Positives) 数量: {TP_count}")
print(f"MD (Missed Detections) 数量: {MD_count}")
print(f"FA (False Alarms) 数量: {FA_count}")
print(f"FP 定位错误 (IoU < 0.5 且类别预测正确) 数量: {loc_error_count}")
print(f"FP 分类错误 (IoU > 0.5 且类别预测错误) 数量: {cls_error_count}")
