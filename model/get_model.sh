wget -nc --directory-prefix=./ 		http://posefs1.perception.cs.cmu.edu/Users/ZheCao/pose_iter_440000.caffemodel
python -m caffe2.python.caffe_translator pose_deploy.prototxt pose_iter_440000.caffemodel
