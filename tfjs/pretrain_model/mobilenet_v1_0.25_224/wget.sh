#!/bin/bash
for i in {1..55}
do
   wget  https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/group${i}-shard1of1
done
