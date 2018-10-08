//index.js
//获取应用实例
const app = getApp()

// 获取图像RGB数据
var getImageRGB = function (canvasId, imgUrl, callback, imgWidth, imgHeight) {
  console.log("entering getBase64Image");
  const ctx = wx.createCanvasContext(canvasId);
  ctx.drawImage(imgUrl, 0, 0, imgWidth || 299, imgHeight || 299);
  ctx.draw(false, () => {
    console.log("ctx.draw");
    // API 1.9.0 获取图像数据
    wx.canvasGetImageData({
      canvasId: canvasId,
      x: 0,
      y: 0,
      width: imgWidth || 299,
      height: imgHeight || 299,
      success(res) {
        var result = res;
        console.log([result.data.buffer]);
        
        var i, j;
        rows = [];
        for (i = 0; i < result.width; i++) {
          cols = [];
          for (j = 0; j < result.height; j++) {
            rgb = [];
            index = i * result.width + j * 4;           // 每个点包含RGBA 4个分量
            rgb.push(result.data[index] / 255);       // r
            rgb.push(result.data[index + 1] / 255);   // g
            rgb.push(result.data[index + 2] / 255);   // b
            // 忽略alpha值

            cols.push(rgb);
          }
          rows.push(cols);
        }
        
        callback(rows);
      },
      fail: e => {
        console.error(e);
      },
    })
  })
};


Page({
  data: {
    imgUrl: '',
  },
  onLoad: function () {
    console.log("onLoad");
  },

  // 从相册选择
  doChooseImage: function () {
    console.log("doChooseImage");

    // 选择图片
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['album'],
      success: function (res) {
        console.log("wx.chooseImage success")
        wx.showLoading({
          title: '处理中',
        })

        const filePath = res.tempFilePaths[0]
        console.log("filePath:" + filePath);
        // this.setData({
        //   imgUrl: filePath
        // }) //将生成的图片url保存下来，后面继续处理

        getImageRGB('dogCanvas', filePath, function (rgbData) {
          //  在此处得到的RGB数据
          console.log("getImageRGB");
          json_data = {
            "model_name": "default", "data": { "image": [] }
          }
          json_data["data"]["image"] = [rgbData];
          console.log(json_data);

          wx.request({
            url: "https://ilego.club:8500",
            // header: {
            //   "Content-Type": "application/x-www-form-urlencoded"
            // },
            method: "POST",
            data: json_data,
            success: function (respose) {
              prediction = respose.data["prediction"];
              console.log(prediction);
            }
          });
        });
      },
      fail: e => {
        console.error(e);
      },
      complete: function () {
        console.log("complete");
        wx.hideLoading();
      }
    })
  },
  // 拍照
  doTakePhoto: function () {
    // 选择图片
    wx.chooseImage({
      count: 1,
      sizeType: ['compressed'],
      sourceType: ['camera'],
      success: function (res) {
      },
      fail: e => {
        console.error(e);
      }
    })
  },
})
