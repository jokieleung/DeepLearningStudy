max pooling��CNN���е����ֵ�ػ���������ʵ�÷��;��������
��Щ�ط����ԴӾ��ȥ�ο���TensorFlow��tf.nn.conv2d������ʵ�־���ģ� 
tf.nn.max_pool(value, ksize, strides, padding, name=None)

�������ĸ����;�������ƣ�
��һ������value����Ҫ�ػ������룬һ��ػ�����ھ������棬��������ͨ����feature map����Ȼ��[batch, height, width, channels]������shape
�ڶ�������ksize���ػ����ڵĴ�С��ȡһ����ά������һ����[1, height, width, 1]����Ϊ���ǲ�����batch��channels�����ػ�������������ά����Ϊ��1
����������strides���;�����ƣ�������ÿһ��ά���ϻ����Ĳ�����һ��Ҳ��[1, stride,stride, 1]
���ĸ�����padding���;�����ƣ�����ȡ'VALID' ����'SAME'
����һ��Tensor�����Ͳ��䣬shape��Ȼ��[batch, height, width, channels]������ʽ
