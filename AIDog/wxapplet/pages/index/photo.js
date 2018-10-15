// pages/index/photo.js

//引入本地json数据
var dogs_json = require('../../utils/dogs_data.js');

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
        console.log("buf:" + [result.data.buffer]);

        var i, j;
        var rows = [];
        for (i = 0; i < result.width; i++) {
          var cols = [];
          for (j = 0; j < result.height; j++) {
            var rgb = [];
            var index = i * result.width + j * 4;         // 每个点包含RGBA 4个分量
            rgb.push(result.data[index] / 255);       // r
            rgb.push(result.data[index + 1] / 255);   // g
            rgb.push(result.data[index + 2] / 255);   // b
            // 忽略alpha值

            cols.push(rgb);
          }
          rows.push(cols);
        }

        console.log("rows:" + rows);
        callback(rows);
      },
      fail: e => {
        console.error(e);
      },
    })
  })
};

Page({

  /**
   * 页面的初始数据
   */
  data: {
    imgUrl: '',
    dogList: {},
  },

  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {
    this.setData({
      //dogs_json.dog_list获取dogs_data.js里定义的json数据，并赋值给dogList
      dogList: dogs_json.dog_list,
      imgUrl: options.filePath
    });
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {
    wx.showLoading({
      title: '正在识别中...',
    });
  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
    var filePath = this.data.imgUrl;
    var that = this;

    getImageRGB('dogCanvas', filePath, function (rgbData) {
      //  在此处得到的RGB数据
      console.log("getImageRGB");
      var json_data = {
        "model_name": "default", "data": { "image": [] }
      }
      json_data["data"]["image"] = [rgbData];
      console.log("json_data:" + json_data);

      wx.request({
        url: "https://ilego.club:8500",
        // header: {
        //   "Content-Type": "application/x-www-form-urlencoded"
        // },
        method: "POST",
        data: json_data,
        success: function (response) {
          console.log("wx.request success!")
          var prediction = response.data["prediction"];
          console.log("response:" + response)
          console.log(prediction);
          var max = 0;
          var index = 0;
          for (var i = 0; i < prediction[0].length; i++) {
            console.log(i + ":" + prediction[0][i])
            if (prediction[i] > max) {
              max = prediction[i];
              index = i;
            }
          }
          console.log("max:" + max + ", index:" + index);
          if (max > 0.1) {
            var dogInfo = that.data.dogList[index];
            console.log(dogInfo);
            that.setData({
              found: true,
              cname: dogInfo["cname"],
              ename: dogInfo["ename"],
              description: dogInfo["description"],
            });
          } else {
            that.setData({
              found: false,
            })
          }
          console.log("hideLoading");
          wx.hideLoading();
        },
        fail: e => {
          console.error(e);
          console.log("hideLoading");
          wx.hideLoading();
        },
        complete: function() {
          console.log("complete");
        }
      });
    });
  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})