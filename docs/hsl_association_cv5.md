<!-- 用途：说明共享HGNN关联加权架构及本次标准五折运行的固定协议。 -->
# 关联加权反馈与标准CV5

所有项目代码编写、修改、检查和训练仅在/root/TRY服务器执行。

原候选H0经原BBP选择得到m；投影节点经共享的第一层HGNN(H0)和ReLU得到上下文，层归一化后的节点、超边均值及二者乘积/差值输入4h→h→1关联MLP。仅计算已有STC成员对。s=0.2*tanh(MLP)，w=|e|*exp(s)/sum_members(exp(s))，H1=m⊙H0⊙w。末层零初始化使w=1。每条边关联支持与总质量保持不变，User/Mention权重为1。第一版不删成员、不新增关联、不分解大边、不加对比损失。

H1用于完整两层HGNN和超边特征聚合。IBG的Jaccard及Louvain拓扑仅读取m⊙H0的二值前向值，保留原选择器直通梯度路径。原BBP选择、覆盖核、尾部约束和预算策略不改；正式CV5恢复联合训练，不固定父边掩码。raw/context/none为显式开关；本次只运行context模型一次CV5，不宣称已完成raw匹配容量消融。

沿用configs/vine_hsl_server.ini中的归档超参数，另存configs/vine_association_cv5_server.ini：
- VINE可信数据SHA256=C50F3263CA28D156953414D9E4AD1CCF48FF0045A7867880636F3A2E99339548。
- seed42划出固定194最终会话；776开发会话按StratifiedKFold(5,shuffle=True,random_state=42)轮换。
- 每折训练20轮、batch64、Adam lr1e-4、weight_decay0.005、hidden128、dropout0.35、正类权重1.5、原始置信度加权。
- robust_adaptive，coverage_q0.2、tail_gamma0.85、eta0.25、保留比例边界0到1；训练Gumbel启用，验证禁用。不是此前固定70%的短训练。
- 全部折同一训练种子42。各折只用自己的训练会话计算类别和置信度统计。
- 按五折验证positive_f1均值选唯一共同epoch；平局取标准差较小者、再取较早epoch，复用原标准入口函数。
- 每轮验证预测与checkpoint落盘。输出共同epoch五折均值/样本标准差、完整776条OOF预测及合并指标；补充固定第20轮结果。此为开发CV评估，选epoch后指标非嵌套无偏测试估计；不评估最终194会话。
- 无标签确定性原始图缓存仅减少重复构图；逐批记录编号修正，测试核对缓存与原构图完全一致。未缓存训练表示或梯度。

运行示例：
PYTHONPATH=. python scripts/train_vine_cv5.py --config configs/vine_association_cv5_server.ini --mode bbp --architecture association --association-mode context --association-alpha 0.2 --cache-graphs --torch-threads 4 --seed 42 --run-name UNIQUE_NAME

标准运行由scripts/run_association_cv5.py保存源码快照、环境、命令、预检及结果审计。验证后原模型默认入口仍为original架构。历史归档基线若环境不同只能作为参考，不据此宣称同环境因果增益。

五折由独立进程并发共享同一GPU，使用一份校验元信息的原始图缓存；每折随机状态独立，全部完成后统一聚合选epoch。并发仅改变调度，未改变CV划分、训练超参数和验证规则。耗时不是独占GPU基准。可选最终测试分支独立构图，不读取仅含开发集的缓存。
