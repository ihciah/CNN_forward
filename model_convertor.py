#!/usr/bin/env python
# coding:utf-8
#
# Copyright 2015 By ihciah
# https://github.com/ihciah/CNN_forward
#
import caffe,struct
import numpy as np
#from PIL import Image


class Convertor:
    def __init__(self,output,net):
        self.output=open(output,"wb+",0)
        self.net=net

    def write_data(self,layer_id,layer_name):
        d=self.net.layers[layer_id]
        if d.type=='Convolution':
            shape=list(d.blobs[0].data.shape) #conv
            allcount=1
            for i in shape:
                allcount*=i
            layer_name=list(layer_name[:15])
            for i in range(16-len(layer_name)):
                layer_name.append("\0")
            type='c'
            print layer_name
            self.output.write(struct.pack("16c", *layer_name))
            self.output.write(struct.pack("c", *type))
            self.output.write(struct.pack("4i", *shape))
            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[0].data.flatten()))

            shape=list(d.blobs[1].data.shape) #bias
            allcount=1
            for i in shape:
                allcount*=i

            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[1].data.flatten()))
            self.output.write(struct.pack("i",0))
        if d.type=='InnerProduct':
            shape=list(d.blobs[0].data.shape) #dense
            allcount=1
            for i in shape:
                allcount*=i
            layer_name=list(layer_name[:15])
            for i in range(16-len(layer_name)):
                layer_name.append("\0")
            type='d'
            print layer_name
            self.output.write(struct.pack("16c", *layer_name))
            self.output.write(struct.pack("c", *type))
            self.output.write(struct.pack("2i", *shape))
            self.output.write(struct.pack("i", 1))
            self.output.write(struct.pack("i", 1))
            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[0].data.flatten()))

            shape=list(d.blobs[1].data.shape) #bias
            allcount=1
            for i in shape:
                allcount*=i

            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[1].data.flatten()))
            self.output.write(struct.pack("i",0))



DEPLOY_PROTOTXT="/home/ch/workspace/forward-test/type2/deploy.prototxt"
TRAINED_NET="/home/ch/workspace/forward-test/new__iter_10000.caffemodel"

net = caffe.Classifier(DEPLOY_PROTOTXT,TRAINED_NET)
print net
print list(net._layer_names)
print net.layers[3].blobs[0].data.shape
conv=Convertor("test_output",net)
conv.write_data(0,"conv21")
conv.write_data(3,"conv22")
conv.write_data(6,"conv23")
conv.write_data(10,"ip2a")
conv.write_data(13,"ipfinala")

#src="/home/ch/workspace/forward-test/1.jpg"
#img=np.array(Image.open(src).convert('L'))
#net.blobs['data'].data[0]=img/255.0
#out = net.forward()
#print 1
