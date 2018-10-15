### Tensorflow 示例

一些展现Tensorflow用法的示例代码

#### 代码说明

* saved_model

示例说明SavedModel格式模型的保存与加载。

  * save_model.py - 保存一个简单的手写识别模型到当前目录的./model下
  * load_model.py - 加载save_model.py所保存的位于./model下的模型，并进行预测。
  * export_logdir.py - 读取模型，并导出TensorBoard所需的logdir
  * connect_model.py - 连接两个Tensorflow模型，组成一个新的模型并保存
  * test_connect_model.py - 测试connect_model.py的新模型

