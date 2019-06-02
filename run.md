# 运行方法
nohup ./bin/submit.sh &

# 结果目录
在output文件夹会生成csv文件 和 相应的模型

# 备注

- 模型的生成

运行时间大于1小时左右, 会直接download 基础模型进行迁移训练

- Inpu 文件夹结构

./input/train_face_value_label.csv

./input/train_data/*.jpg

./input/public_test_data/*.jpg
