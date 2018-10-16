### AIDog

一款从图片识别狗的类别的应用，包括Android版和微信小程序版。

#### 源码说明

 * data
   
   包含狗的类别信息的数据及处理脚本，数据收集自百度百科和维基百科。
   
   - dogs.xls - Office Excel格式的数据
   - dogs.csv - CSV格式的数据
   - csv_to_json.py - CSV格式转换为JSON格式的脚本，在微信小程序和Android程序中都使用JSON格式的数据
 
 * serving
  
   包含重新训练狗类别识别模型的脚本，以及为了部署而对模型进行重建的脚本。
   
   * retrain.py - 从Inception V3模型重新训练狗类别识别模型的脚本
   * rebuild_model.py - 为了给微信小程序提供RESTful API，对retrain模型做了重建，使其接受base64字符串形式的图片数据。
   * test_rebuild_model.py - 测试rebuild的模型，直接inference模型
   * test_client.py - 用来测试服务器上所部署模型的简单测试客户端，适用于rebuild的模型
   * test_client_v1.py - 测试客户端的最初版本，适用于retrain出来的模型
   * dog_labels_inception_v3.txt - 狗类别标签列表，这个列表是在retrain过程中生成的
 
 * wxapplet
 
   包含微信小程序的源码。
  
#### 相关文档

    