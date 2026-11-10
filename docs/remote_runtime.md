<!-- 用途：记录已实际准备完成的服务器实验环境及项目运行位置，不保存密码。 -->
# 服务器运行状态
2026-11-09 已成功在服务器执行类 HSL 训练。SSH 主机 hn01-ssh.gpuhome.cc，端口 30963，用户 root；项目 /root/TRY。
Python /opt/conda/bin/python，RTX 4090 24GB，PyTorch 2.12.1+cu130，CUDA 可用。
已补齐 torch-geometric、scikit-learn、python-louvain。具体安装版本见输出日志或 environment.json。
VINE 快照 /root/TRY/data/vine/KbertConfidence/processed/KbertConfidence_data_bert.pt 的 SHA256 与归档 C50F3263CA28D156953414D9E4AD1CCF48FF0045A7867880636F3A2E99339548 相符。
原配置不覆盖；实验使用 configs/vine_hsl_server.ini，设备 cuda:0。
本地仅用于 SSH 通道；后续源文件修改与所有项目验证均在此服务器完成。
实验日志使用服务器默认时间；不将其当作 Asia/Shanghai 本地时间。
