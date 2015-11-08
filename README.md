# CNN_forward
#### A [Convolution Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network) forward code for [caffe](http://caffe.berkeleyvision.org/) implemented in C++.

* Features
  * Load model converted from `*.caffemodel`, model encrypt is supported.
  * Define layers' topology simply.
  * Using Intel's [TBB](https://www.threadingbuildingblocks.org/) which makes convolution faster on multicore CPU.
  * Supported layers currently: `INPUT`, `CONVOLUTION`, `POOLING`, `DENSE`(or `INNER_PRODUCT`), `RELU`.
  * Easy to be compiled into a single `.exe`, `.dll`, or `.so`,it can be executed without any additional library.

* How to use
  * In `main.cpp` there is a example:
    1. First initialize a `CnnNet` object `net`, and then call `net.init('model','key')`, which will load the model named `model` and the blowfish key is `key`.
    2. Then call `net.forward('test.jpg',GRAY)`, which will read the file `test.jpg` in `GRAY` mode and do the net forward.
    3. Finally you can get the result and process it by yourself, or use `net.argmax()`. The function `argmax` is not really a argmax, and its result is not between 0 to 1, in fact, it will fetch all layers' max values whose `output` is defined as `true` and return them in vector.
    4. Since this example does a captcha recognition job, I call a simple function in `utils.cpp` to convert the numbers in the vector above to letters.
  * Define net's topology.
    1. In `CnnNet.cpp`, we can define net in `CnnNet::init`. Just `new` a `LayerConfig` and push_back it's address.
    2. If the `INPUT` layer's size(w,h) is set, all images will be resized when doing forward. Leave blank or set to `0` means pass the resize process.
    3. Some information is read from model, and here we don't need to define them, for example, the kernel size of convolution.
    4. When you don't set a layer's parent, it will be set to its previously pushed layer. If you want to set it, you can use string or vector to set it.
* Todo jobs
  * Make it faster and faster, maybe support GPU.
  * Separate the net's weights and the images calculated to make it threadsafe.
  * Add more layer support, such as LRN.
  * Zip the model file.
* Copyright
  * This it NOT an open source software. DONOT distribute this code. ONLY used by yourself and pay attention to anti-decompile.
